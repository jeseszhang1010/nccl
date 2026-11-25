/*************************************************************************
 * SPDX-License-Identifier: Apache-2.0
 *
 * MRC GIN device backend, CPU RQ mode (NCCL_GIN_MRC_GPU_RQ=0). Included by
 * gin_mrc.h, which describes both modes.
 *
 * A signal is the immediate of a Write-with-Immediate, posted with
 * doca_gpu_dev_verbs_put_immediate(): the data and the signal are a single
 * operation, so the signal is ordered behind its payload. The immediate packs
 * the signal op, id and value (DOCA_GPUNETIO_VERBS_SIGNAL_IMM_*); a
 * ncclGin_SignalAdd value that does not fit is split into several adds. The
 * receive queues are managed by the CPU: the host plugin drains them into a
 * per-context signal map, which adds each immediate's value to the GPU slot
 * registered for its id (doca_gpu_verbs_rq_progress() semantics).
 * Device code only reads signal slots; a reset records a baseline (see
 * ncclGinMrcGPUContext::signal_base). A signal without data (a bare signal, a
 * further chunk of a large Add, the RDW get response) is a zero-length
 * Write-with-Immediate, which carries no data segment.
 *
 * Get: the requester writes a request descriptor into the data owner's
 * request ring with a Write-with-Immediate that adds 1 to the owner's request
 * counter for it; the owner's device progress serves it with RDMA Writes back
 * to the requester followed by a Write-with-Immediate that adds 1 to the
 * requester's OK or error response counter.
 *
 * Device code comes from the open-source DOCA GPUNetIO headers NCCL installs
 * from DOCA_HOME, shared with the GDAKI backend.
 *************************************************************************/

#ifndef _NCCL_DEVICE_GIN_MRC_CPU_RQ_H_
#define _NCCL_DEVICE_GIN_MRC_CPU_RQ_H_

// Same configuration as the GDAKI backend, which includes the same headers.
#ifndef DOCA_GPUNETIO_USE_CUDA_WRAPPER
#define DOCA_GPUNETIO_USE_CUDA_WRAPPER
#endif
#ifndef DOCA_VERBS_USE_NET_WRAPPER
#define DOCA_VERBS_USE_NET_WRAPPER
#endif

#include "../gdaki/doca_gpunetio/doca_gpunetio_device.h"

namespace nccl {
namespace gin {
namespace mrc {

static constexpr auto kNicHandler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB;
// The plugin creates every send CQ as plain 64 B CQEs in GPU memory.
static constexpr auto kCqType = DOCA_GPUNETIO_VERBS_CQ_64B;
static constexpr auto kCtrlCqUpdate = DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE;

NCCL_DEVICE_INLINE static ncclGinMrcGPUContext* getCtx(ncclGinCtx ctx) {
  return &((struct ncclGinMrcGPUContext*)ctx.handle)[ctx.contextId];
}

NCCL_DEVICE_INLINE static doca_gpu_dev_verbs_qp* getQp(ncclGinMrcGPUContext* mrc, int peer) {
  return (doca_gpu_dev_verbs_qp*)nccl::utility::loadConst(&mrc->qps) + peer;
}

// Signal slots are written by the CPU through a GDRCopy mapping, so read them at system scope.
NCCL_DEVICE_INLINE static uint64_t loadSlot(uint64_t* slot) {
  return cuda::atomic_ref<uint64_t, cuda::thread_scope_system>(*slot).load(cuda::memory_order_acquire);
}

/* ── Signal immediates ────────────────────────────────────────────── */

// Largest value one immediate adds, and the most immediates a single ncclGin_SignalAdd may use.
static constexpr uint64_t kSigValMax = DOCA_GPUNETIO_VERBS_SIGNAL_IMM_VAL_MASK;
static constexpr uint32_t kSigMaxImms = 64;

// A signal to deliver: n Write-with-Immediate adds to signal id, each of at most kSigValMax, the
// last one carrying the remainder. Adds commute, so the receiver needs no reassembly.
struct ncclGinMrcSignal {
  uint32_t n; // number of immediates (0 = no signal)
  uint32_t id;
  uint64_t value;
};

NCCL_DEVICE_INLINE static ncclGinMrcSignal encodeSignal(bool hasSignal, uint32_t signalId,
                                                        ncclGinSignalOp_t signalOp, uint64_t signalOpArg) {
  ncclGinMrcSignal s = {0, signalId, 0};
  if (!hasSignal) return s;
  assert(signalId <= DOCA_GPUNETIO_VERBS_SIGNAL_IMM_ID_MASK && "MRC GIN: signal id exceeds the signal map id space");
  s.value = signalOp == ncclGinSignalInc ? 1 : signalOpArg;
  // An add of 0 still sends one immediate, so the signal keeps its ordering with the payload.
  s.n = s.value <= kSigValMax ? 1 : (uint32_t)((s.value + kSigValMax - 1) / kSigValMax);
  assert(s.n <= kSigMaxImms && "MRC GIN: ncclGin_SignalAdd value too large");
  return s;
}

NCCL_DEVICE_INLINE static uint32_t signalImm(const ncclGinMrcSignal& s, uint32_t k) {
  const uint64_t v = k + 1 < s.n ? kSigValMax : s.value - (uint64_t)(s.n - 1) * kSigValMax;
  return doca_gpu_dev_verbs_pack_signal_imm<DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD>(s.id, (uint16_t)v);
}

NCCL_DEVICE_INLINE static uint32_t addOneImm(uint32_t signalId) {
  return doca_gpu_dev_verbs_pack_signal_imm<DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD>(signalId, 1);
}

/* ── RDW (Receiver-Driven Write) get protocol ────────────────────── */

static constexpr uint32_t kRdwDescBytes = NCCL_GIN_MRC_RDW_DESC_BYTES;
static constexpr uint32_t kRdwMaxSubs = 4;
static constexpr uint32_t kRdwMaxSubBytes = 8u << 20;

struct ncclGinMrcGetSub {
  uint32_t src_lo, src_hi; // offset in the server's window (zero-based MR)
  uint32_t dst_lo, dst_hi; // offset in the requester's window (zero-based MR)
  uint32_t len;
  uint32_t src_key; // big-endian server rkey, translated to an lkey through the server's MR table
  uint32_t dst_key; // big-endian requester rkey
};
struct ncclGinMrcGetDesc {
  uint32_t tag;  // request id + 1, so a stale or overwritten slot is detectable
  uint32_t nsub; // 1..kRdwMaxSubs
  uint32_t reserved[2];
  struct ncclGinMrcGetSub sub[kRdwMaxSubs];
};
static_assert(sizeof(ncclGinMrcGetDesc) == kRdwDescBytes, "RDW descriptor must be 128 B");

NCCL_DEVICE_INLINE static uint32_t rdwTag(uint64_t id) {
  return (uint32_t)id + 1;
}

// Byte offsets inside one context's region of the RDW MR (layout in gin_mrc_dev.h).
NCCL_DEVICE_INLINE static uint64_t rdwRingOffset(uint32_t window, int requester, uint32_t slot) {
  return ((uint64_t)requester * window + slot) * kRdwDescBytes;
}
NCCL_DEVICE_INLINE static uint64_t rdwStagingOffset(int nRanks, uint32_t window, int peer, uint32_t slot) {
  return ((uint64_t)nRanks * window + (uint64_t)peer * window + slot) * kRdwDescBytes;
}

// Signal ids of the RDW counters (layout in gin_mrc_dev.h).
NCCL_DEVICE_INLINE static uint32_t rdwRequestId(ncclGinMrcGPUContext* mrc, int requester) {
  return nccl::utility::loadConst(&mrc->rdw_signal_base) + requester;
}
NCCL_DEVICE_INLINE static uint32_t rdwResponseId(ncclGinMrcGPUContext* mrc, int nRanks, int server, bool ok) {
  return nccl::utility::loadConst(&mrc->rdw_signal_base) + (ok ? 1 : 2) * nRanks + server;
}

// Responses received from peer so far, and how many of them reported an error.
NCCL_DEVICE_INLINE static uint64_t rdwCompleted(ncclGinCtx ctx, ncclGinMrcGPUContext* mrc, int peer,
                                                uint64_t* errors = nullptr) {
  uint64_t* signals = nccl::utility::loadConst(&mrc->signals);
  const uint64_t err = loadSlot(signals + rdwResponseId(mrc, ctx.nRanks, peer, false));
  if (errors) *errors = err;
  return loadSlot(signals + rdwResponseId(mrc, ctx.nRanks, peer, true)) + err;
}

/* ── Send path ────────────────────────────────────────────────────── */

// Record that the operation whose last WQE index is wqeIdx must advance counter counterId
// (increment by 1) once it completes locally. The SQ-CQ drainer applies it. The slot is keyed
// by the WQE index so concurrent posters write disjoint slots without coordination; the paired
// counter_pending bump gates the drainer so counter-free traffic pays nothing. Must be called
// before the doorbell is rung so the slot store is ordered ahead of the completion. No-op when
// the context has no counter table installed.
NCCL_DEVICE_INLINE static void recordCounter(ncclGinMrcPeerState* state, uint64_t wqeIdx, uint32_t counterId) {
  if (state == nullptr || state->counter_slots == nullptr) return;
  uint32_t slot = (uint32_t)wqeIdx & state->sq_wqe_mask;
  cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(state->counter_slots[slot])
    .store(counterId + 1, cuda::memory_order_release);
  cuda::atomic_ref<unsigned int, cuda::thread_scope_device>(state->counter_pending)
    .fetch_add(1, cuda::memory_order_relaxed);
}

// Remote target of a zero-length write to peer: the NIC accesses no memory, but the WQE still names
// a valid key, the peer's RDW MR (registered for every context).
NCCL_DEVICE_INLINE static doca_gpu_dev_verbs_addr bareTarget(ncclGinMrcGPUContext* mrc, int peer) {
  using nccl::utility::loadConst;
  doca_gpu_dev_verbs_addr raddr;
  raddr.addr = loadConst(&mrc->rdw_ctx_offset);
  raddr.key = loadConst(loadConst(&mrc->rdw_rkeys) + peer);
  return raddr;
}

// Posts a (possibly chunked) RDMA Write; when useImm is set the last chunk is an RDMA
// Write-with-Immediate carrying imm (a packed signal immediate). bytes == 0 posts one zero-length
// WQE without a data segment (laddr is unused). Must be called by a single thread; the QP uses a
// GPU-SM doorbell so the submit rings the NIC directly. Returns the last reserved WQE index.
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode>
NCCL_DEVICE_INLINE static uint64_t write(doca_gpu_dev_verbs_qp* qp, doca_gpu_dev_verbs_addr raddr,
                                         doca_gpu_dev_verbs_addr laddr, size_t bytes, bool useImm, uint32_t imm) {
  doca_gpu_dev_verbs_ticket_t ticket;
  if (useImm)
    doca_gpu_dev_verbs_put_immediate<DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD, resource_sharing_mode, kNicHandler, kCqType>(
      qp, raddr, laddr, bytes,
      (imm >> DOCA_GPUNETIO_VERBS_SIGNAL_IMM_ID_SHIFT) & DOCA_GPUNETIO_VERBS_SIGNAL_IMM_ID_MASK,
      (uint16_t)((imm >> DOCA_GPUNETIO_VERBS_SIGNAL_IMM_VAL_SHIFT) & DOCA_GPUNETIO_VERBS_SIGNAL_IMM_VAL_MASK), &ticket);
  else
    doca_gpu_dev_verbs_put<resource_sharing_mode, kNicHandler, DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD, kCqType>(
      qp, raddr, laddr, bytes, &ticket);
  return ticket;
}

// General signaled write, for what put_immediate cannot express in one call: an optional data
// write followed by the signal's immediates in one WQE reservation, and/or a counter. The first
// immediate rides the last data WQE when there is data; every other immediate (and a
// counter-only op without data) is a zero-length write. When counterState is non-null the last WQE
// index is recorded for send-CQ counter draining before the doorbell. Must be called by a single
// thread. Returns the last reserved WQE index.
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode>
NCCL_DEVICE_INLINE static uint64_t writeSignaled(ncclGinMrcGPUContext* mrc, doca_gpu_dev_verbs_qp* qp, int peer,
                                                 doca_gpu_dev_verbs_addr raddr, doca_gpu_dev_verbs_addr laddr,
                                                 size_t bytes, const ncclGinMrcSignal& sig,
                                                 ncclGinMrcPeerState* counterState, uint32_t counterId) {
  using nccl::utility::loadConst;
  const uint64_t num_data =
    (bytes == 0) ? 0 : doca_gpu_dev_verbs_div_ceil_aligned_pow2(bytes, DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE_SHIFT);
  uint32_t num_bare = (num_data > 0) ? (sig.n > 0 ? sig.n - 1 : 0) : sig.n;
  if (num_data == 0 && num_bare == 0) num_bare = 1; // counter-only op without data
  const uint64_t base_wqe_idx = doca_gpu_dev_verbs_reserve_wq_slots<resource_sharing_mode, DOCA_GPUNETIO_VERBS_QP_SQ,
                                                                    kCqType>(qp, num_data + num_bare);
  uint64_t wqe_idx = base_wqe_idx;

  size_t remaining = bytes;
  for (uint64_t i = 0; i < num_data; i++) {
    wqe_idx = base_wqe_idx + i;
    const size_t size_ =
      remaining > DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE ? DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE : remaining;
    const bool immHere = (i + 1 == num_data) && sig.n > 0;
    doca_gpu_dev_verbs_wqe_prepare_write(
      qp, doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx), wqe_idx,
      immHere ? DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE_IMM : DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE, kCtrlCqUpdate,
      immHere ? signalImm(sig, 0) : 0, raddr.addr + (i * DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE), raddr.key,
      laddr.addr + (i * DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE), laddr.key, size_);
    remaining -= size_;
  }

  // Zero-length WQEs: with 0 bytes prepare_write sets two segments, so the NIC reads no data segment.
  const doca_gpu_dev_verbs_addr bare = bareTarget(mrc, peer);
  const uint32_t firstBareImm = (num_data > 0) ? 1 : 0;
  for (uint32_t k = 0; k < num_bare; k++) {
    wqe_idx = base_wqe_idx + num_data + k;
    const bool hasImm = firstBareImm + k < sig.n;
    doca_gpu_dev_verbs_wqe_prepare_write(
      qp, doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx), wqe_idx,
      hasImm ? DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE_IMM : DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE, kCtrlCqUpdate,
      hasImm ? signalImm(sig, firstBareImm + k) : 0, bare.addr, bare.key, 0, 0, 0);
  }

  doca_gpu_dev_verbs_mark_wqes_ready<resource_sharing_mode>(qp, base_wqe_idx, wqe_idx);
  // Record the counter before submit: submit's release fence then orders the slot store ahead of
  // the doorbell, so the SQ-CQ drainer can never observe the completion before the slot is set.
  if (counterState != nullptr) recordCounter(counterState, wqe_idx, counterId);
  doca_gpu_dev_verbs_submit<resource_sharing_mode, DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU, kNicHandler>(qp, wqe_idx + 1);
  return wqe_idx;
}

// Non-blocking send-CQ drain that advances counters for completed put-with-counter operations
// on one peer QP. Walks the SQ CQ from sq_cons_index while counter records are outstanding;
// each completed WQE that carries a recorded counterId bumps counters[counterId] by 1.
// Caller holds the peer's drain lock.
NCCL_DEVICE_INLINE static void sqDrainCounters(doca_gpu_dev_verbs_qp* qp, ncclGinMrcPeerState* state,
                                               uint64_t* countersBuf) {
  if (state->counter_slots == nullptr || state->sq_status != 0) return;
  while (cuda::atomic_ref<unsigned int, cuda::thread_scope_device>(state->counter_pending)
           .load(cuda::memory_order_relaxed) != 0) {
    const uint64_t idx = state->sq_cons_index;
    const int status = doca_gpu_dev_verbs_poll_one_cq_at<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
                                                         DOCA_GPUNETIO_VERBS_QP_SQ, kCqType>(qp, idx);
    if (status == EBUSY) return;
    if (status != 0) {
      state->sq_status = status;
      return;
    }
    const uint32_t slot = (uint32_t)idx & state->sq_wqe_mask;
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> rec(state->counter_slots[slot]);
    const uint32_t v = rec.load(cuda::memory_order_acquire);
    if (v != 0) {
      rec.store(0, cuda::memory_order_relaxed);
      cuda::atomic_ref<uint64_t, cuda::thread_scope_device>(countersBuf[v - 1])
        .fetch_add(1, cuda::memory_order_relaxed);
      cuda::atomic_ref<unsigned int, cuda::thread_scope_device>(state->counter_pending)
        .fetch_sub(1, cuda::memory_order_relaxed);
    }
    state->sq_cons_index = idx + 1;
  }
}

/* ── RDW get: server side (run by device progress) ────────────────── */

// Translate a requester-supplied source rkey into the local lkey of the registered window that
// owns it, checking that [off, off+len) lies inside that window. The host updates the table while
// gets may be in flight (an entry is published by its rkey or by count, and removed entries are
// tombstoned with rkey 0 / size 0), so read it through volatile to see each update.
NCCL_DEVICE_INLINE static bool rdwLookupSource(ncclGinMrcMrTable* table, uint32_t key, uint64_t off, uint32_t len,
                                               uint32_t* lkey) {
  if (table == nullptr) return false;
  volatile ncclGinMrcMrTable* t = table;
  uint32_t n = t->count;
  if (n > NCCL_GIN_MRC_MR_TABLE_CAPACITY) n = NCCL_GIN_MRC_MR_TABLE_CAPACITY;
  for (uint32_t i = 0; i < n; i++) {
    if (t->entries[i].rkey != key) continue;
    const uint64_t size = t->entries[i].size;
    if (off > size || len > size - off) return false;
    *lkey = t->entries[i].lkey;
    return true;
  }
  return false;
}

// Serve request number `id` from `requester`: copy the descriptor out of the request ring,
// validate it, write each sub-request's data back, then answer with a Write-with-Immediate that
// adds 1 to the requester's OK or error response counter for this rank. MRC places the earlier
// Writes on this QP before the response completes, so the requester sees the data once its
// response counter covers the request. Caller holds the requester's drain lock.
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode>
NCCL_DEVICE_INLINE static void rdwServe(ncclGinCtx ctx, ncclGinMrcGPUContext* mrc, doca_gpu_dev_verbs_qp* qp,
                                        int requester, uint64_t id) {
  using nccl::utility::loadConst;
  const uint32_t window = loadConst(&mrc->rdw_window);
  const uint32_t slot = (uint32_t)id & (window - 1);
  uint32_t* ring = (uint32_t*)(loadConst(&mrc->rdw_region) + rdwRingOffset(window, requester, slot));

  // The NIC wrote the slot; read it at system scope so no stale L1 copy from an earlier request
  // in the same slot is used.
  ncclGinMrcGetDesc desc;
  uint32_t* words = (uint32_t*)&desc;
  for (uint32_t i = 0; i < kRdwDescBytes / sizeof(uint32_t); i++)
    words[i] = doca_gpu_dev_verbs_load_relaxed_sys_global(ring + i);

  bool ok = desc.tag == rdwTag(id) && desc.nsub > 0 && desc.nsub <= kRdwMaxSubs;
  uint32_t lkeys[kRdwMaxSubs];
  if (ok) {
    ncclGinMrcMrTable* table = loadConst(&mrc->mr_table);
    for (uint32_t s = 0; s < desc.nsub && ok; s++) {
      const ncclGinMrcGetSub& sub = desc.sub[s];
      const uint64_t srcOff = ((uint64_t)sub.src_hi << 32) | sub.src_lo;
      ok = sub.len != 0 && sub.len <= kRdwMaxSubBytes && rdwLookupSource(table, sub.src_key, srcOff, sub.len, &lkeys[s]);
    }
  }

  if (ok) {
    for (uint32_t s = 0; s < desc.nsub; s++) {
      const ncclGinMrcGetSub& sub = desc.sub[s];
      doca_gpu_dev_verbs_addr raddr, laddr;
      raddr.addr = ((uint64_t)sub.dst_hi << 32) | sub.dst_lo;
      raddr.key = sub.dst_key;
      laddr.addr = ((uint64_t)sub.src_hi << 32) | sub.src_lo;
      laddr.key = lkeys[s];
      write<resource_sharing_mode>(qp, raddr, laddr, sub.len, /*useImm=*/false, 0);
    }
  }

  // The response is a zero-length Write-with-Immediate: its immediate is all the requester needs.
  const doca_gpu_dev_verbs_addr none = {};
  write<resource_sharing_mode>(qp, bareTarget(mrc, requester), none, 0, /*useImm=*/true,
                               addOneImm(rdwResponseId(mrc, ctx.nRanks, ctx.rank, ok)));
}

NCCL_DEVICE_INLINE static void rdwServeDispatch(ncclGinCtx ctx, ncclGinMrcGPUContext* mrc, doca_gpu_dev_verbs_qp* qp,
                                                int requester, uint64_t id) {
  switch ((ncclGinResourceSharingMode)ctx.resourceSharingMode) {
  case NCCL_GIN_RESOURCE_SHARING_THREAD:
    rdwServe<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(ctx, mrc, qp, requester, id);
    break;
  case NCCL_GIN_RESOURCE_SHARING_CTA:
    rdwServe<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA>(ctx, mrc, qp, requester, id);
    break;
  default:
    rdwServe<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(ctx, mrc, qp, requester, id);
    break;
  }
}

/* ── Device progress ──────────────────────────────────────────────── */

// Signals need no device progress: the host plugin delivers them into the signal slots. What is
// left is local work: advance counters by draining each peer's send CQ, and serve the RDW get
// requests the peers' request counters announce. Non-blocking; called from the signal/counter
// wait loops so a waiting thread makes progress on its own context.
//
// Many CTAs share one GIN context, so this is called concurrently. Each peer's work runs under a
// per-peer try-lock; a thread that fails to acquire skips that peer this round, and every waiter
// keeps calling this from its spin loop, so nothing is missed and no thread starves. Peers with
// nothing pending are skipped without taking the lock.
NCCL_DEVICE_INLINE static void progress(ncclGinCtx ctx) {
  using nccl::utility::loadConst;
  ncclGinMrcGPUContext* mrc = getCtx(ctx);
  ncclGinMrcPeerState* peers = loadConst(&mrc->peers);
  if (peers == nullptr) return;
  uint64_t* signals = loadConst(&mrc->signals);
  uint64_t* counterBuf = loadConst(&mrc->counters);
  const bool rdwOn = loadConst(&mrc->rdw_window) != 0;
  for (int peer = 0; peer < ctx.nRanks; peer++) {
    ncclGinMrcPeerState* state = peers + peer;
    cuda::atomic_ref<uint64_t, cuda::thread_scope_device> served(state->rdw_served);
    // Gets from self are local writes and never send requests.
    const uint64_t requests = (rdwOn && peer != ctx.rank) ? loadSlot(signals + rdwRequestId(mrc, peer)) : 0;
    const bool rdwPending = requests != served.load(cuda::memory_order_relaxed);
    const bool counterPending = state->counter_slots != nullptr &&
                                cuda::atomic_ref<unsigned int, cuda::thread_scope_device>(state->counter_pending)
                                    .load(cuda::memory_order_relaxed) != 0;
    if (!rdwPending && !counterPending) continue;

    cuda::atomic_ref<int, cuda::thread_scope_device> drainLock(state->drain_lock);
    int unlocked = 0;
    if (!drainLock.compare_exchange_strong(unlocked, 1, cuda::memory_order_acquire, cuda::memory_order_relaxed))
      continue;
    doca_gpu_dev_verbs_qp* qp = getQp(mrc, peer);
    if (rdwPending) {
      // The counts read above may be stale: another thread may have served them since. Both only
      // grow, so re-read the request count under the lock and serve strictly below it.
      const uint64_t arrived = loadSlot(signals + rdwRequestId(mrc, peer));
      for (uint64_t id = served.load(cuda::memory_order_relaxed); id < arrived; id++) {
        rdwServeDispatch(ctx, mrc, qp, peer, id);
        served.store(id + 1, cuda::memory_order_relaxed);
      }
    }
    if (counterPending) sqDrainCounters(qp, state, counterBuf);
    drainLock.store(0, cuda::memory_order_release);
  }
}

/* ── Put / PutValue ───────────────────────────────────────────────── */

// A put with a single-immediate signal and no counter is one doca_gpu_dev_verbs_put_immediate()
// (a zero-length one for a bare signal); a data-only put is a plain RDMA Write. Puts with counters
// or multi-immediate signals go through writeSignaled. A zero-byte put is sent like a bare signal,
// and is a no-op when it carries neither a signal nor a counter. Like data puts, a bare signal does
// not wait for its completion: later reservations (slot availability) and flush() reap it.
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode, typename Coop>
NCCL_DEVICE_INLINE static void putImplMode(ncclGinCtx ctx, Coop coop, int peer, bool hasWins,
                                           ncclGinWindow_t dstWin, size_t dstOff, ncclGinWindow_t srcWin, size_t srcOff,
                                           size_t bytes, const ncclGinMrcSignal& sig, bool hasCounter,
                                           uint32_t counterId, cuda::thread_scope required, cuda::thread_scope given,
                                           uint32_t optFlags) {
  using nccl::utility::loadConst;
  (void)optFlags;
  coop.sync();
  const bool hasData = hasWins && bytes > 0;
  if (coop.thread_rank() == 0 && (hasData || sig.n > 0 || hasCounter)) {
    ncclGinMrcGPUContext* mrc = getCtx(ctx);
    doca_gpu_dev_verbs_qp* qp = getQp(mrc, peer);

    if ((required == cuda::thread_scope_system) && (given > required)) {
      doca_gpu_dev_verbs_fence_release<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
    }

    doca_gpu_dev_verbs_addr raddr = {}, laddr = {};
    if (hasData) {
      raddr.addr = dstOff;
      raddr.key = loadConst(loadConst(&((ncclGinMrcMemHandle*)dstWin)->rkeys) + peer);
      laddr.addr = srcOff;
      laddr.key = loadConst(&((ncclGinMrcMemHandle*)srcWin)->lkey);
    }
    if (!hasCounter && sig.n <= 1) {
      if (!hasData) raddr = bareTarget(mrc, peer);
      write<resource_sharing_mode>(qp, raddr, laddr, hasData ? bytes : 0, sig.n == 1,
                                   sig.n == 1 ? signalImm(sig, 0) : 0);
    } else {
      // Counter is advanced by the send-CQ drainer keyed off this peer's state.
      ncclGinMrcPeerState* counterState = hasCounter ? (loadConst(&mrc->peers) + peer) : nullptr;
      writeSignaled<resource_sharing_mode>(mrc, qp, peer, raddr, laddr, hasData ? bytes : 0, sig, counterState,
                                           counterId);
    }
  }
  coop.sync();
}

template <typename Coop>
NCCL_DEVICE_INLINE static void putImpl(ncclGinCtx ctx, Coop coop, int peer, bool hasWins, ncclGinWindow_t dstWin,
                                       size_t dstOff, ncclGinWindow_t srcWin, size_t srcOff, size_t bytes,
                                       const ncclGinMrcSignal& sig, bool hasCounter, uint32_t counterId,
                                       cuda::thread_scope required, cuda::thread_scope given, uint32_t optFlags) {
  switch ((ncclGinResourceSharingMode)ctx.resourceSharingMode) {
  case NCCL_GIN_RESOURCE_SHARING_THREAD:
    putImplMode<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(ctx, coop, peer, hasWins, dstWin, dstOff, srcWin,
                                                                     srcOff, bytes, sig, hasCounter, counterId,
                                                                     required, given, optFlags);
    break;
  case NCCL_GIN_RESOURCE_SHARING_CTA:
    putImplMode<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA>(ctx, coop, peer, hasWins, dstWin, dstOff, srcWin, srcOff,
                                                               bytes, sig, hasCounter, counterId, required, given,
                                                               optFlags);
    break;
  default:
    putImplMode<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(ctx, coop, peer, hasWins, dstWin, dstOff, srcWin, srcOff,
                                                               bytes, sig, hasCounter, counterId, required, given,
                                                               optFlags);
    break;
  }
}

// The small value is delivered by an inline RDMA Write; when a signal is requested, following
// zero-length Write-with-Immediates carry the signal immediates. Same-QP ordering guarantees the
// value lands before the notification is observed.
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode, typename Coop, typename T>
NCCL_DEVICE_INLINE static void putValueImplMode(ncclGinCtx ctx, Coop coop, int peer, ncclGinWindow_t dstWin,
                                                size_t dstOff, T srcData, const ncclGinMrcSignal& sig,
                                                cuda::thread_scope required, cuda::thread_scope given,
                                                uint32_t optFlags) {
  using nccl::utility::loadConst;
  (void)optFlags;
  coop.sync();
  if (coop.thread_rank() == 0) {
    ncclGinMrcGPUContext* mrc = getCtx(ctx);
    doca_gpu_dev_verbs_qp* qp = getQp(mrc, peer);
    ncclGinMrcMemHandle* dstMh = (ncclGinMrcMemHandle*)dstWin;
    doca_gpu_dev_verbs_addr raddr;
    raddr.addr = dstOff;
    raddr.key = loadConst(loadConst(&dstMh->rkeys) + peer);
    if ((required == cuda::thread_scope_system) && (given > required)) {
      doca_gpu_dev_verbs_fence_release<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
    }
    // The generic doca_gpu_dev_verbs_p switch instantiates a warp variant; this backend always
    // posts from a single thread, so call the thread variant directly.
    doca_gpu_dev_verbs_ticket_t ticket;
    doca_gpu_dev_verbs_p_thread<T, resource_sharing_mode, kNicHandler, kCqType>(qp, raddr, srcData, &ticket);
    if (sig.n > 0) {
      doca_gpu_dev_verbs_addr none = {};
      writeSignaled<resource_sharing_mode>(mrc, qp, peer, none, none, 0, sig, nullptr, 0);
    }
  }
  coop.sync();
}

template <typename Coop, typename T>
NCCL_DEVICE_INLINE static void putValueImpl(ncclGinCtx ctx, Coop coop, int peer, ncclGinWindow_t dstWin, size_t dstOff,
                                            T srcData, const ncclGinMrcSignal& sig, cuda::thread_scope required,
                                            cuda::thread_scope given, uint32_t optFlags) {
  switch ((ncclGinResourceSharingMode)ctx.resourceSharingMode) {
  case NCCL_GIN_RESOURCE_SHARING_THREAD:
    putValueImplMode<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(ctx, coop, peer, dstWin, dstOff, srcData, sig,
                                                                          required, given, optFlags);
    break;
  case NCCL_GIN_RESOURCE_SHARING_CTA:
    putValueImplMode<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA>(ctx, coop, peer, dstWin, dstOff, srcData, sig,
                                                                    required, given, optFlags);
    break;
  default:
    putValueImplMode<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(ctx, coop, peer, dstWin, dstOff, srcData, sig,
                                                                    required, given, optFlags);
    break;
  }
}

/* ── Get (RDW requester side) ─────────────────────────────────────── */

// Post RDW requests for `bytes` from the peer's window (srcKey/srcOff) into the local window
// (dstKey/dstOff): up to kRdwMaxSubs sub-requests of at most kRdwMaxSubBytes per request, each
// request one 128 B Write-with-Immediate into the peer's request ring slot id % window. Posting
// is serialized per peer under get_lock so request ids are issued in wire order; at most
// `window` requests are in flight, and a full window is waited out while driving progress
// (which also serves the peer's own requests to this rank, so two ranks fetching from each other
// cannot starve one another).
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode>
NCCL_DEVICE_INLINE static void rdwPostGets(ncclGinCtx ctx, ncclGinMrcGPUContext* mrc, doca_gpu_dev_verbs_qp* qp,
                                           int peer, uint32_t srcKey, uint64_t srcOff, uint32_t dstKey,
                                           uint64_t dstOff, size_t bytes) {
  using nccl::utility::loadConst;
  ncclGinMrcPeerState* state = loadConst(&mrc->peers) + peer;
  const uint32_t window = loadConst(&mrc->rdw_window);
  uint8_t* region = loadConst(&mrc->rdw_region);
  const uint64_t ctxOffset = loadConst(&mrc->rdw_ctx_offset);
  const uint32_t peerRkey = loadConst(loadConst(&mrc->rdw_rkeys) + peer);
  const uint32_t rdwLkey = loadConst(&mrc->rdw_lkey);
  const uint32_t requestImm = addOneImm(rdwRequestId(mrc, ctx.rank));

  cuda::atomic_ref<int, cuda::thread_scope_device> lock(state->get_lock);
  int unlocked = 0;
  while (!lock.compare_exchange_weak(unlocked, 1, cuda::memory_order_acquire, cuda::memory_order_relaxed))
    unlocked = 0;

  cuda::atomic_ref<uint64_t, cuda::thread_scope_device> issued(state->get_issued);
  uint64_t id = issued.load(cuda::memory_order_relaxed);
  size_t done = 0;
  while (done < bytes) {
    NVCC_PRAGMA_UNROLL_DISABLED
    while (id - rdwCompleted(ctx, mrc, peer) >= window) progress(ctx);

    const uint32_t slot = (uint32_t)id & (window - 1);
    ncclGinMrcGetDesc* desc = (ncclGinMrcGetDesc*)(region + rdwStagingOffset(ctx.nRanks, window, peer, slot));
    uint32_t nsub = 0;
    for (; nsub < kRdwMaxSubs && done < bytes; nsub++) {
      const uint32_t len = (bytes - done) > kRdwMaxSubBytes ? kRdwMaxSubBytes : (uint32_t)(bytes - done);
      const uint64_t s = srcOff + done;
      const uint64_t d = dstOff + done;
      desc->sub[nsub] = {(uint32_t)s, (uint32_t)(s >> 32), (uint32_t)d, (uint32_t)(d >> 32), len, srcKey, dstKey};
      done += len;
    }
    desc->tag = rdwTag(id);
    desc->nsub = nsub;
    // The NIC reads the descriptor from GPU memory once the doorbell rings.
    __threadfence_system();

    doca_gpu_dev_verbs_addr raddr, laddr;
    raddr.addr = ctxOffset + rdwRingOffset(window, ctx.rank, slot);
    raddr.key = peerRkey;
    laddr.addr = ctxOffset + rdwStagingOffset(ctx.nRanks, window, peer, slot);
    laddr.key = rdwLkey;
    write<resource_sharing_mode>(qp, raddr, laddr, kRdwDescBytes, /*useImm=*/true, requestImm);
    issued.store(++id, cuda::memory_order_release);
  }
  lock.store(0, cuda::memory_order_release);
}

template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode, typename Coop>
NCCL_DEVICE_INLINE static void getImplMode(ncclGinCtx ctx, Coop coop, int peer, ncclGinWindow_t remoteWin,
                                           size_t remoteOff, ncclGinWindow_t localWin, size_t localOff, size_t bytes) {
  using nccl::utility::loadConst;
  coop.sync();
  if (coop.thread_rank() == 0 && bytes > 0) {
    ncclGinMrcGPUContext* mrc = getCtx(ctx);
    doca_gpu_dev_verbs_qp* qp = getQp(mrc, peer);
    ncclGinMrcMemHandle* remoteMh = (ncclGinMrcMemHandle*)remoteWin;
    ncclGinMrcMemHandle* localMh = (ncclGinMrcMemHandle*)localWin;
    const uint32_t dstKey = loadConst(loadConst(&localMh->rkeys) + ctx.rank);
    if (peer == ctx.rank) {
      // Loopback: the self QP is connected to a local responder, so a plain Write from the
      // source window into the destination window is the get. The send CQ covers completion.
      doca_gpu_dev_verbs_addr raddr, laddr;
      raddr.addr = localOff;
      raddr.key = dstKey;
      laddr.addr = remoteOff;
      laddr.key = loadConst(&remoteMh->lkey);
      write<resource_sharing_mode>(qp, raddr, laddr, bytes, /*useImm=*/false, 0);
    } else {
      rdwPostGets<resource_sharing_mode>(ctx, mrc, qp, peer, loadConst(loadConst(&remoteMh->rkeys) + peer),
                                         remoteOff, dstKey, localOff, bytes);
    }
  }
  coop.sync();
}

template <typename Coop>
NCCL_DEVICE_INLINE static void getImpl(ncclGinCtx ctx, Coop coop, int peer, ncclGinWindow_t remoteWin,
                                       size_t remoteOff, ncclGinWindow_t localWin, size_t localOff, size_t bytes) {
  switch ((ncclGinResourceSharingMode)ctx.resourceSharingMode) {
  case NCCL_GIN_RESOURCE_SHARING_THREAD:
    getImplMode<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(ctx, coop, peer, remoteWin, remoteOff, localWin,
                                                                     localOff, bytes);
    break;
  case NCCL_GIN_RESOURCE_SHARING_CTA:
    getImplMode<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA>(ctx, coop, peer, remoteWin, remoteOff, localWin,
                                                               localOff, bytes);
    break;
  default:
    getImplMode<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(ctx, coop, peer, remoteWin, remoteOff, localWin,
                                                               localOff, bytes);
    break;
  }
}

// Wait until every get to this peer issued before `ticket` (low 32 bits of get_issued) has been
// answered, driving progress while spinning. Returns ncclRemoteError if any answer was an error.
template <bool HasTimeout>
NCCL_DEVICE_INLINE static ncclResult_t rdwWaitGets(ncclGinCtx ctx, ncclGinMrcGPUContext* mrc, int peer,
                                                   uint32_t ticket, uint32_t* abortFlag, uint64_t startCycle,
                                                   uint64_t timeoutCycles, uint32_t& steps) {
  (void)startCycle;
  (void)timeoutCycles;
  uint64_t errors = 0;
  NVCC_PRAGMA_UNROLL_DISABLED
  while ((int32_t)((uint32_t)rdwCompleted(ctx, mrc, peer, &errors) - ticket) < 0) {
    progress(ctx);
    if NCCL_IF_CONSTEXPR (HasTimeout) {
      if (clock64() - startCycle >= timeoutCycles) return ncclTimeout;
    }
    if (nccl::utility::testAbort(abortFlag, steps)) return ncclSuccess;
  }
  return errors ? ncclRemoteError : ncclSuccess;
}

// The void flush/wait variants have no way to return an error, and the fetched data cannot be
// trusted, so fail loudly.
NCCL_DEVICE_INLINE static void rdwFailGets(ncclGinCtx ctx) {
  printf("NCCL GIN/MRC: rank %d context %d: get failed (RDW error response)\n", ctx.rank, ctx.contextId);
  __trap();
}

/* ── Completion: flush / wait ─────────────────────────────────────── */

// The flush watermark is the current reservation index on the peer's QP, plus, for gets, the
// number of RDW requests issued to the peer: get data is only visible once their responses have
// been delivered.
template <enum doca_gpu_dev_verbs_resource_sharing_mode doca_sharing_mode>
NCCL_DEVICE_INLINE static void flushAsyncImpl(ncclGinCtx ctx, int peer, ncclGinRequest_t* outRequest) {
  ncclGinMrcRequest* req = reinterpret_cast<ncclGinMrcRequest*>(outRequest);
  ncclGinMrcGPUContext* mrc = getCtx(ctx);
  req->peer = peer;
  doca_gpu_dev_verbs_qp* qp = getQp(mrc, peer);
  req->sq_rsvd_index = doca_gpu_dev_verbs_atomic_read<uint64_t, doca_sharing_mode>(&qp->sq_rsvd_index);
  ncclGinMrcPeerState* peers = nccl::utility::loadConst(&mrc->peers);
  req->get_ticket = peers ? (uint32_t)cuda::atomic_ref<uint64_t, cuda::thread_scope_device>(peers[peer].get_issued)
                              .load(cuda::memory_order_acquire)
                          : 0;
}

template <bool HasTimeout, enum doca_gpu_dev_verbs_resource_sharing_mode doca_sharing_mode>
NCCL_DEVICE_INLINE static ncclResult_t waitImplCore(ncclGinCtx ctx, ncclGinRequest_t& request, cuda::memory_order ord,
                                                    uint32_t* abortFlag, uint64_t timeoutCycles) {
  using nccl::utility::testAbort;
  (void)timeoutCycles; // referenced only when HasTimeout is true
  ncclGinMrcRequest& req = reinterpret_cast<ncclGinMrcRequest&>(request);
  ncclGinMrcGPUContext* mrc = getCtx(ctx);
  doca_gpu_dev_verbs_qp* qp = getQp(mrc, req.peer);
  uint32_t steps = 0;
  uint64_t startCycle = 0;
  (void)startCycle; // referenced only when HasTimeout is true
  if NCCL_IF_CONSTEXPR (HasTimeout) startCycle = clock64();

  if (req.sq_rsvd_index != 0) {
    const doca_gpu_dev_verbs_ticket_t pollIdx = static_cast<doca_gpu_dev_verbs_ticket_t>(req.sq_rsvd_index - 1u);
    bool done = false;
    // doca_gpu_dev_verbs_wait blocks indefinitely, so it can only be used when the caller has no
    // deadline AND no abort flag to honor.
    if NCCL_IF_CONSTEXPR (!HasTimeout) {
      if (abortFlag == nullptr) {
        doca_gpu_dev_verbs_wait<doca_sharing_mode, kNicHandler, kCqType>(qp, pollIdx);
        done = true;
      }
    }
    NVCC_PRAGMA_UNROLL_DISABLED
    while (!done) {
      int status =
        doca_gpu_dev_verbs_poll_one_cq_at<doca_sharing_mode, DOCA_GPUNETIO_VERBS_QP_SQ, kCqType>(qp, pollIdx);
      if (status == 0) break;
      if NCCL_IF_CONSTEXPR (HasTimeout) {
        if (clock64() - startCycle >= timeoutCycles) return ncclTimeout;
      }
      if (testAbort(abortFlag, steps)) {
        cuda::atomic_thread_fence(ord, cuda::thread_scope_system);
        return ncclSuccess;
      }
    }
  }

  if (nccl::utility::loadConst(&mrc->peers) != nullptr && req.peer != ctx.rank) {
    ncclResult_t ret =
      rdwWaitGets<HasTimeout>(ctx, mrc, req.peer, req.get_ticket, abortFlag, startCycle, timeoutCycles, steps);
    if (ret != ncclSuccess) return ret;
  }
  cuda::atomic_thread_fence(ord, cuda::thread_scope_system);
  return ncclSuccess;
}

template <enum doca_gpu_dev_verbs_resource_sharing_mode doca_sharing_mode>
NCCL_DEVICE_INLINE static void waitImpl(ncclGinCtx ctx, ncclGinRequest_t& request, cuda::memory_order ord,
                                        uint32_t* abortFlag) {
  if (waitImplCore</*HasTimeout=*/false, doca_sharing_mode>(ctx, request, ord, abortFlag, 0) == ncclRemoteError)
    rdwFailGets(ctx);
}

template <enum doca_gpu_dev_verbs_resource_sharing_mode doca_sharing_mode>
NCCL_DEVICE_INLINE static ncclResult_t waitImpl(ncclGinCtx ctx, ncclGinRequest_t& request, cuda::memory_order ord,
                                                uint32_t* abortFlag, uint64_t timeoutCycles) {
  return waitImplCore</*HasTimeout=*/true, doca_sharing_mode>(ctx, request, ord, abortFlag, timeoutCycles);
}

template <bool HasTimeout, enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode, typename Coop>
NCCL_DEVICE_INLINE static ncclResult_t flushImplModeCore(ncclGinCtx ctx, Coop coop, cuda::memory_order ord,
                                                         uint32_t* abortFlag, uint64_t timeoutCycles) {
  (void)timeoutCycles; // referenced only when HasTimeout is true
  using nccl::utility::loadConst;
  using nccl::utility::testAbort;

  ncclGinMrcGPUContext* mrc = getCtx(ctx);
  doca_gpu_dev_verbs_qp* qps = (doca_gpu_dev_verbs_qp*)loadConst(&mrc->qps);
  ncclGinMrcPeerState* peers = loadConst(&mrc->peers);
  uint32_t steps = 0;
  uint64_t startCycle = 0;
  (void)startCycle; // referenced only when HasTimeout is true
  if NCCL_IF_CONSTEXPR (HasTimeout) startCycle = clock64();

  NVCC_PRAGMA_UNROLL_DISABLED
  for (int peer = coop.thread_rank(); peer < ctx.nRanks; peer += coop.size()) {
    uint64_t ticket = doca_gpu_dev_verbs_atomic_read<uint64_t, resource_sharing_mode>(&qps[peer].sq_rsvd_index);
    // No need to poll the send CQ if there are no outstanding operations.
    bool done = ticket == 0;
    if (!done) --ticket;

    if NCCL_IF_CONSTEXPR (!HasTimeout) {
      if (!done && abortFlag == nullptr) {
        doca_gpu_dev_verbs_wait<resource_sharing_mode, kNicHandler, kCqType>(&qps[peer], ticket);
        done = true;
      }
    }

    NVCC_PRAGMA_UNROLL_DISABLED
    while (!done) {
      int status = doca_gpu_dev_verbs_poll_one_cq_at<resource_sharing_mode, DOCA_GPUNETIO_VERBS_QP_SQ, kCqType>(
        &qps[peer], ticket);
      if (status == 0) break;
      if NCCL_IF_CONSTEXPR (HasTimeout) {
        if (clock64() - startCycle >= timeoutCycles) return ncclTimeout;
      }
      if (testAbort(abortFlag, steps)) {
        if NCCL_IF_CONSTEXPR (HasTimeout) {
          cuda::atomic_thread_fence(ord, cuda::thread_scope_system);
          return ncclSuccess;
        } else break; // Non-timeout path advances to the next peer on abort.
      }
    }

    // Gets from this peer are only locally visible once their RDW responses are delivered.
    // Loopback gets are plain Writes covered by the send-CQ wait above.
    if (peers != nullptr && peer != ctx.rank) {
      const uint32_t issued = (uint32_t)cuda::atomic_ref<uint64_t, cuda::thread_scope_device>(peers[peer].get_issued)
                                .load(cuda::memory_order_acquire);
      ncclResult_t ret = rdwWaitGets<HasTimeout>(ctx, mrc, peer, issued, abortFlag, startCycle, timeoutCycles, steps);
      if (ret == ncclTimeout) return ret;
      if (ret == ncclRemoteError) {
        if NCCL_IF_CONSTEXPR (HasTimeout) return ret;
        else rdwFailGets(ctx);
      }
    }
  }
  cuda::atomic_thread_fence(ord, cuda::thread_scope_system);
  return ncclSuccess;
}

template <typename Coop>
NCCL_DEVICE_INLINE static ncclResult_t flushImpl(ncclGinCtx ctx, Coop coop, cuda::memory_order ord, uint32_t* abortFlag,
                                                 uint64_t timeoutCycles) {
  switch ((ncclGinResourceSharingMode)ctx.resourceSharingMode) {
  case NCCL_GIN_RESOURCE_SHARING_THREAD:
    return flushImplModeCore<true, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(ctx, coop, ord, abortFlag,
                                                                                        timeoutCycles);
  case NCCL_GIN_RESOURCE_SHARING_CTA:
    return flushImplModeCore<true, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA>(ctx, coop, ord, abortFlag,
                                                                                  timeoutCycles);
  default:
    return flushImplModeCore<true, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(ctx, coop, ord, abortFlag,
                                                                                  timeoutCycles);
  }
}

template <typename Coop>
NCCL_DEVICE_INLINE static void flushImpl(ncclGinCtx ctx, Coop coop, cuda::memory_order ord, uint32_t* abortFlag) {
  switch ((ncclGinResourceSharingMode)ctx.resourceSharingMode) {
  case NCCL_GIN_RESOURCE_SHARING_THREAD:
    (void)flushImplModeCore<false, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(ctx, coop, ord, abortFlag, 0);
    break;
  case NCCL_GIN_RESOURCE_SHARING_CTA:
    (void)flushImplModeCore<false, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA>(ctx, coop, ord, abortFlag, 0);
    break;
  default:
    (void)flushImplModeCore<false, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(ctx, coop, ord, abortFlag, 0);
    break;
  }
}

/* ── Signal slots ─────────────────────────────────────────────────── */

// Signal slots belong to the host signal map, which keeps its own running total of each slot and
// would overwrite a device-side store, so a reset records the current total as the signal's
// baseline instead.
NCCL_DEVICE_INLINE static void resetSignal(ncclGinCtx ctx, uint32_t signalId) {
  ncclGinMrcGPUContext* mrc = getCtx(ctx);
  nccl::utility::loadConst(&mrc->signal_base)[signalId] = loadSlot(nccl::utility::loadConst(&mrc->signals) + signalId);
}

// The signal's logical value is its slot minus the baseline recorded by its last reset.
NCCL_DEVICE_INLINE static ncclGinOffsetPtr signalPtr(ncclGinCtx ctx, uint32_t signalId) {
  ncclGinMrcGPUContext* mrc = getCtx(ctx);
  return {nccl::utility::loadConst(&mrc->signals) + signalId,
          *(volatile uint64_t*)(nccl::utility::loadConst(&mrc->signal_base) + signalId)};
}

} // namespace mrc
} // namespace gin
} // namespace nccl

#endif /* _NCCL_DEVICE_GIN_MRC_CPU_RQ_H_ */
