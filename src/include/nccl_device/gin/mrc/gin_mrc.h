/*************************************************************************
 * SPDX-License-Identifier: Apache-2.0
 *
 * MRC implementations of the NCCL GIN device-side API.
 *
 * MRC QPs speak RDMA Write and RDMA Write-with-Immediate only: no atomics, no
 * remote read. Consequences that shape this backend:
 *   - A signal is an inbound Write-with-Immediate carrying the signal id (and,
 *     for SignalAdd, the value, split across several immediates if needed). The
 *     receiver posts receive WQEs and drains its receive CQ to apply them to a
 *     local signal array (see drainSignals / ncclGinApi_Progress).
 *   - A counter is advanced locally by draining the send CQ: a put-with-counter
 *     records its counter id keyed by the last WQE index, and the drain bumps
 *     the counter once that WQE completes.
 *   - Only INDEXED signals are representable; VA signals are unsupported.
 *   - Get uses Receiver-Driven Write (RDW): the requester writes a request descriptor
 *     into the data owner's request ring with a Write-with-Immediate; the owner's
 *     receive drain serves it with RDMA Writes back to the requester followed by a
 *     Write-with-Immediate response on the same QP. Requests are only served while
 *     the data owner runs a kernel that drives GIN progress (any signal/counter
 *     read or wait, flush, or wait on a request).
 *
 * The GPU-resident context is created by the out-of-tree host plugin; its layout
 * is gin_mrc_dev.h.
 *************************************************************************/

#ifndef _NCCL_DEVICE_GIN_MRC_H_
#define _NCCL_DEVICE_GIN_MRC_H_

#include <errno.h>
#include <cassert>
#include <cstdint>

// Bake in what the vendored DOCA device headers expect so applications compiling
// against the installed NCCL headers never have to pass these themselves.
#ifndef DOCA_GPUNETIO_USE_CUDA_WRAPPER
#define DOCA_GPUNETIO_USE_CUDA_WRAPPER
#endif
#ifndef DOCA_VERBS_USE_NET_WRAPPER
#define DOCA_VERBS_USE_NET_WRAPPER
#endif
#ifndef GPUNETIO_VERBS_RCX_SUPPORTED
#define GPUNETIO_VERBS_RCX_SUPPORTED
#endif
#ifndef DOCA_ALLOW_EXPERIMENTAL_API
#define DOCA_ALLOW_EXPERIMENTAL_API
#endif

#include "nccl_device/utility.h"
#include "../gin_device_common.h"
#include "gin_mrc_dev.h"
#include "../../transport/net_mrc/doca-gpunetio-rcx/include/doca_gpunetio_rcx_device.h"

struct ncclGinMrcRequest {
  int peer;
  uint32_t get_ticket; // low 32 bits of the peer's get_issued at flushAsync time
  uint64_t sq_rsvd_index;
};
static_assert(sizeof(ncclGinMrcRequest) <= sizeof(ncclGinRequest_t),
              "ncclGinMrcRequest must fit in ncclGinRequest_t");

namespace nccl {
namespace gin {
namespace mrc {

// The receive WQE SGE is unused for Write-with-Immediate: the payload lands via the RDMA
// write address and the RQE only consumes a notification slot. Post a zero-length segment
// with an intentionally invalid lkey.
static constexpr uint32_t kRecvInvalidLkey = 0x100u;
// mlx5 responder CQE opcode for a completion carrying immediate data.
static constexpr uint32_t kCqeRespWriteImm = 0x1u;

NCCL_DEVICE_INLINE static ncclGinMrcGPUContext* getCtx(ncclGinCtx ctx) {
  return &((struct ncclGinMrcGPUContext*)ctx.handle)[ctx.contextId];
}

NCCL_DEVICE_INLINE static doca_gpu_dev_verbs_qp* getQp(ncclGinMrcGPUContext* mrc, int peer) {
  return (doca_gpu_dev_verbs_qp*)nccl::utility::loadConst(&mrc->qps) + peer;
}

/* ── Signal immediates ────────────────────────────────────────────── */

// Immediate layout of a signal (bit 31 clear; RDW messages set kRdwFlag, see below):
//   Inc:  [30] 0                                      [29:0]  signal id
//   Add:  [30] 1  [29] more  [28:13] signal id  [12:0]  13-bit chunk of the value
// An immediate is the only per-message payload the receiver can attribute, so an Add value
// that does not fit one chunk is split into consecutive "more" chunks, least significant first.
// They are posted in a single WQE reservation, so they reach the peer's receive CQ back to back;
// the drain accumulates them per peer and applies the sum to the signal once, on the last chunk.
static constexpr uint32_t kSigAddFlag = 1u << 30;
static constexpr uint32_t kSigAddMore = 1u << 29;
static constexpr uint32_t kSigAddChunkBits = 13;
static constexpr uint32_t kSigAddChunkMask = (1u << kSigAddChunkBits) - 1;
static constexpr uint32_t kSigAddIdBits = 16;
static constexpr uint32_t kSigAddIdMask = (1u << kSigAddIdBits) - 1;
static constexpr uint32_t kSigAddMaxChunks = (64 + kSigAddChunkBits - 1) / kSigAddChunkBits;

struct ncclGinMrcSignalImms {
  uint32_t n; // number of Write-with-Immediate messages (0 = no signal)
  uint32_t imm[kSigAddMaxChunks];
};

NCCL_DEVICE_INLINE static ncclGinMrcSignalImms encodeSignal(bool hasSignal, uint32_t signalId,
                                                            ncclGinSignalOp_t signalOp, uint64_t signalOpArg) {
  ncclGinMrcSignalImms s;
  s.n = 0;
  if (!hasSignal) return s;
  if (signalOp == ncclGinSignalInc || signalOpArg == 1) {
    assert(signalId < kSigAddFlag && "MRC GIN: signal id too large");
    s.n = 1;
    s.imm[0] = signalId;
    return s;
  }
  assert(signalId <= kSigAddIdMask && "MRC GIN: signal id too large for ncclGin_SignalAdd with a value other than 1");
  do {
    const uint32_t chunk = (uint32_t)signalOpArg & kSigAddChunkMask;
    signalOpArg >>= kSigAddChunkBits;
    s.imm[s.n++] = kSigAddFlag | (signalOpArg ? kSigAddMore : 0u) | ((signalId & kSigAddIdMask) << kSigAddChunkBits) |
                   chunk;
  } while (signalOpArg);
  return s;
}

/* ── RDW (Receiver-Driven Write) get protocol ────────────────────── */

// Immediate layout. Signals carry a signal immediate (bit 31 clear); RDW messages set kRdwFlag:
//   [31] RDW  [30] response  [29:28] status (responses)  [27:22] generation  [21:0] request id
static constexpr uint32_t kRdwFlag = 1u << 31;
static constexpr uint32_t kRdwResp = 1u << 30;
static constexpr uint32_t kRdwStatusShift = 28;
static constexpr uint32_t kRdwGenShift = 22;
static constexpr uint32_t kRdwGenMask = NCCL_GIN_MRC_RDW_GEN_MASK;
static constexpr uint32_t kRdwIdMask = (1u << kRdwGenShift) - 1;

enum : uint32_t {
  kRdwStatusOk = 0,
  kRdwStatusInvalid = 1,  // malformed descriptor or source outside a registered window
  kRdwStatusBusy = 2,     // reserved: the server never answers BUSY yet
  kRdwStatusRejected = 3, // descriptor does not match the request (stale generation or slot)
};

static constexpr uint32_t kRdwDescBytes = NCCL_GIN_MRC_RDW_DESC_BYTES;
static constexpr uint32_t kRdwMaxSubs = 4;
static constexpr uint32_t kRdwMaxSubBytes = 8u << 20;
static constexpr uint32_t kRdwRespBytes = sizeof(uint64_t);

struct ncclGinMrcGetSub {
  uint32_t src_lo, src_hi; // offset in the server's window (zero-based MR)
  uint32_t dst_lo, dst_hi; // offset in the requester's window (zero-based MR)
  uint32_t len;
  uint32_t src_key; // big-endian server rkey, translated to an lkey through the server's MR table
  uint32_t dst_key; // big-endian requester rkey
};
struct ncclGinMrcGetDesc {
  uint32_t tag;  // the request immediate, so a stale or overwritten slot is detectable
  uint32_t nsub; // 1..kRdwMaxSubs
  uint32_t reserved[2];
  struct ncclGinMrcGetSub sub[kRdwMaxSubs];
};
static_assert(sizeof(ncclGinMrcGetDesc) == kRdwDescBytes, "RDW descriptor must be 128 B");

NCCL_DEVICE_INLINE static uint32_t rdwRequestImm(uint32_t gen, uint64_t id) {
  return kRdwFlag | ((gen & kRdwGenMask) << kRdwGenShift) | ((uint32_t)id & kRdwIdMask);
}

NCCL_DEVICE_INLINE static uint32_t rdwResponseImm(uint32_t requestImm, uint32_t status) {
  return (requestImm & ~(3u << kRdwStatusShift)) | kRdwResp | (status << kRdwStatusShift);
}

// Byte offsets inside one context's region of the RDW MR (layout in gin_mrc_dev.h).
NCCL_DEVICE_INLINE static uint64_t rdwRingOffset(uint32_t window, int requester, uint32_t slot) {
  return ((uint64_t)requester * window + slot) * kRdwDescBytes;
}
NCCL_DEVICE_INLINE static uint64_t rdwStagingOffset(int nRanks, uint32_t window, int peer, uint32_t slot) {
  return ((uint64_t)nRanks * window + (uint64_t)peer * window + slot) * kRdwDescBytes;
}
NCCL_DEVICE_INLINE static uint64_t rdwRespOffset(int nRanks, uint32_t window) {
  return 2ull * nRanks * window * kRdwDescBytes;
}

/* ── Receive queue: one-time fill and non-blocking drain ──────────── */

// Regular post-recv of `count` notification WQEs (NCCL_GIN_MRC_RQ_POST_MODE_WQE): reserve the
// RQ slots, write each WQE, mark them ready and submit through the RQ doorbell path, which also
// tracks the RQ producer index. Single-threaded per QP. No availability wait is needed: callers
// only post into slots that are either never used (init) or whose completion was just consumed.
NCCL_DEVICE_INLINE static uint64_t rqPostRecv(struct doca_gpu_dev_verbs_qp* qp, uint32_t count) {
  const uint64_t base = doca_gpu_dev_verbs_reserve_wq_slots<
    DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE, DOCA_GPUNETIO_VERBS_QP_RQ, false>(qp, count);
  for (uint32_t idx = 0; idx < count; idx++) {
    struct mlx5_wqe_data_seg* rwqe = doca_gpu_dev_verbs_get_rwqe_ptr(qp, base + idx);
    doca_gpu_dev_verbs_wqe_prepare_recv(qp, rwqe, 0, doca_gpu_dev_verbs_bswap32(kRecvInvalidLkey), 0);
  }
  doca_gpu_dev_verbs_mark_wqes_ready<
    DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE, DOCA_GPUNETIO_VERBS_QP_RQ>(qp, base, base + count - 1);
  // The exclusive-mode submit does not order the WQE stores ahead of the DBR store.
  doca_gpu_dev_verbs_fence_release<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>();
  // The RQ has no BlueFlame/proxy path: always submit through the DBR, whatever the QP's SQ handler.
  doca_gpu_dev_verbs_submit<
    DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE, DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
    DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB, DOCA_GPUNETIO_VERBS_QP_RQ>(qp, base + count);
  return base;
}

// Fill the whole receive ring with zero-SGE notification WQEs and ring the RQ doorbell. Must be
// called exactly once per QP by a single thread before any peer issues a Write-with-Immediate to
// this QP. counter_slots/counter_capacity install this QP's send-CQ counter-drain table; pass
// counter_slots == nullptr to disable counter support. post_mode (NCCL_GIN_MRC_RQ_POST_MODE_*)
// selects how the ring is posted here and reposted by rqPollOne.
NCCL_DEVICE_INLINE void rqInit(struct doca_gpu_dev_verbs_qp* qp, struct ncclGinMrcRqState* state,
                               uint32_t* counter_slots, uint32_t counter_capacity,
                               uint32_t post_mode = NCCL_GIN_MRC_RQ_POST_MODE_DBR) {
  const uint32_t rqDepth = qp->rq_wqe_num;
  state->consumer_index = 0;
  state->rq_depth = rqDepth;
  state->status = 0;
  state->drain_lock = 0;
  state->post_mode = post_mode == NCCL_GIN_MRC_RQ_POST_MODE_WQE ? NCCL_GIN_MRC_RQ_POST_MODE_WQE
                                                                : NCCL_GIN_MRC_RQ_POST_MODE_DBR;
  state->get_issued = 0;
  state->get_completed = 0;
  state->get_status = 0;
  state->get_lock = 0;
  state->add_acc = 0;
  state->add_shift = 0;

  // The slot table must be a power-of-two at least as large as the SQ ring so that every
  // outstanding WQE index maps to a unique slot; disable counters otherwise.
  state->sq_cons_index = 0;
  state->sq_status = 0;
  state->counter_pending = 0;
  if (counter_slots != nullptr && counter_capacity >= (uint32_t)qp->sq_wqe_num &&
      (counter_capacity & (counter_capacity - 1)) == 0) {
    state->counter_slots = counter_slots;
    state->sq_wqe_mask = counter_capacity - 1;
  } else {
    state->counter_slots = nullptr;
    state->sq_wqe_mask = 0;
  }

  if (state->post_mode == NCCL_GIN_MRC_RQ_POST_MODE_WQE) {
    if (rqPostRecv(qp, rqDepth) != 0) state->status = -1;
    return;
  }

  const uint64_t base = doca_gpu_dev_verbs_reserve_wq_slots<
    DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE, DOCA_GPUNETIO_VERBS_QP_RQ>(qp, rqDepth);
  if (base != 0) {
    state->status = -1;
    return;
  }

  for (uint32_t idx = 0; idx < rqDepth; idx++) {
    struct mlx5_wqe_data_seg* rwqe = doca_gpu_dev_verbs_get_rwqe_ptr(qp, base + idx);
    doca_gpu_dev_verbs_wqe_prepare_recv(qp, rwqe, 0, doca_gpu_dev_verbs_bswap32(kRecvInvalidLkey), 0);
  }

  doca_gpu_dev_verbs_mark_wqes_ready<
    DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE, DOCA_GPUNETIO_VERBS_QP_RQ>(qp, base, base + rqDepth - 1);
  doca_gpu_dev_verbs_update_dbr<
    DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU, DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT,
    DOCA_GPUNETIO_VERBS_QP_RQ>(qp, rqDepth);
}

// Non-blocking drain of a single receive completion. Returns 0 and fills *imm/*opcode when a
// CQE was consumed (and replenishes the RQ slot), EBUSY when none is pending, or a negative
// errno when the CQE reports an error (also latched in state->status).
//
// Not thread-safe: callers must serialize per QP (drainSignals holds the per-peer drain_lock).
NCCL_DEVICE_INLINE static int rqPollOne(struct doca_gpu_dev_verbs_qp* qp, struct ncclGinMrcRqState* state,
                                        uint32_t* imm, uint32_t* opcode) {
  if (state->status != 0) return state->status;

  struct doca_gpu_dev_verbs_cq* cq = doca_gpu_dev_verbs_qp_get_cq_rq(qp);
  const uint64_t ticket = state->consumer_index;
  const int status = doca_gpu_dev_verbs_poll_one_cq_at<
    DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE, DOCA_GPUNETIO_VERBS_QP_RQ>(cq, ticket);
  if (status == EBUSY) return EBUSY;
  if (status != 0) {
    state->status = status;
    return status;
  }

  struct mlx5_cqe64* cqe = &((struct mlx5_cqe64*)cq->cqe_daddr)[ticket & cq->cqe_mask];
  *imm = doca_gpu_dev_verbs_bswap32(
    doca_gpu_dev_verbs_load_relaxed_sys_global((uint32_t*)&cqe->imm_inval_pkey));
  *opcode = doca_gpu_dev_verbs_load_relaxed_sys_global((uint8_t*)&cqe->op_own) >>
            DOCA_GPUNETIO_VERBS_MLX5_CQE_OPCODE_SHIFT;

  state->consumer_index = ticket + 1;
  if (state->post_mode == NCCL_GIN_MRC_RQ_POST_MODE_WQE) {
    // Regular repost: rewrite the just-consumed slot's WQE and submit it.
    rqPostRecv(qp, 1);
    return 0;
  }
  // Replenish the just-consumed immutable receive WQE by advancing only the RQ producer DBR.
  doca_gpu_dev_verbs_update_dbr<
    DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU, DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT,
    DOCA_GPUNETIO_VERBS_QP_RQ>(qp, (uint32_t)(state->consumer_index + state->rq_depth));
  return 0;
}

/* ── Send path ────────────────────────────────────────────────────── */

// Record that the operation whose last WQE index is wqeIdx must advance counter counterId
// (increment by 1) once it completes locally. The SQ-CQ drainer applies it. The slot is keyed
// by the WQE index so concurrent posters write disjoint slots without coordination; the paired
// counter_pending bump gates the drainer so counter-free traffic pays nothing. Must be called
// before the doorbell is rung so the slot store is ordered ahead of the completion. No-op when
// the context has no counter table installed.
NCCL_DEVICE_INLINE static void recordCounter(ncclGinMrcRqState* state, uint64_t wqeIdx, uint32_t counterId) {
  if (state == nullptr || state->counter_slots == nullptr) return;
  uint32_t slot = (uint32_t)wqeIdx & state->sq_wqe_mask;
  cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(state->counter_slots[slot])
    .store(counterId + 1, cuda::memory_order_release);
  cuda::atomic_ref<unsigned int, cuda::thread_scope_device>(state->counter_pending)
    .fetch_add(1, cuda::memory_order_relaxed);
}

// Posts a (possibly chunked) RDMA Write; when useImm is set the last chunk is an RDMA
// Write-with-Immediate carrying signalId. bytes must be non-zero: MRC rejects zero-length
// Write-with-Immediate, so callers send bare signals through the sink scratch instead (see
// writeSignaled). When counterState is non-null the last WQE index
// is recorded for send-CQ counter draining before the doorbell rings. Must be called by a
// single thread; the QP uses a GPU-SM doorbell so the submit rings the NIC directly.
// Returns the last reserved WQE index.
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode>
NCCL_DEVICE_INLINE static uint64_t write(doca_gpu_dev_verbs_qp* qp, doca_gpu_dev_verbs_addr raddr,
                                         doca_gpu_dev_verbs_addr laddr, size_t bytes, bool useImm, uint32_t signalId,
                                         ncclGinMrcRqState* counterState, uint32_t counterId) {
  const uint32_t immBe = doca_gpu_dev_verbs_bswap32(signalId);
  const uint64_t num_chunks =
    (bytes == 0) ? 1
                 : doca_gpu_dev_verbs_div_ceil_aligned_pow2(bytes, DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE_SHIFT);
  const uint64_t base_wqe_idx = doca_gpu_dev_verbs_reserve_wq_slots<resource_sharing_mode>(qp, num_chunks);
  uint64_t wqe_idx = base_wqe_idx;
  size_t remaining = bytes;
  for (uint64_t i = 0; i < num_chunks; i++) {
    wqe_idx = base_wqe_idx + i;
    const size_t size_ =
      remaining > DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE ? DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE : remaining;
    const bool last = (i + 1 == num_chunks);
    doca_gpu_dev_verbs_wqe* wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx);
    doca_gpu_dev_verbs_wqe_prepare_write(
      qp, wqe_ptr, wqe_idx,
      (last && useImm) ? DOCA_GPUNETIO_MLX5_OPCODE_RDMA_WRITE_IMM : DOCA_GPUNETIO_MLX5_OPCODE_RDMA_WRITE,
      DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE, (last && useImm) ? immBe : 0,
      raddr.addr + (i * DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE), raddr.key,
      laddr.addr + (i * DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE), laddr.key, size_);
    remaining -= size_;
  }
  doca_gpu_dev_verbs_mark_wqes_ready<resource_sharing_mode>(qp, base_wqe_idx, wqe_idx);
  // Record the counter before submit: submit's release fence then orders the slot store ahead of
  // the doorbell, so the SQ-CQ drainer can never observe the completion before the slot is set.
  if (counterState != nullptr) recordCounter(counterState, wqe_idx, counterId);
  doca_gpu_dev_verbs_submit<resource_sharing_mode, DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
                            DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB>(qp, wqe_idx + 1);
  return wqe_idx;
}

// Posts an optional data write followed by the signal immediates in sig, in one WQE reservation
// so a multi-chunk Add reaches the peer contiguously. The first immediate rides the last data WQE
// when there is data; every other immediate (and a counter-only op without data) is an 8-byte
// write of the local sink scratch into the peer's reserved sink slot, because MRC rejects
// zero-length writes. When counterState is non-null the last WQE index is recorded for send-CQ
// counter draining. Must be called by a single thread. Returns the last reserved WQE index.
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode>
NCCL_DEVICE_INLINE static uint64_t writeSignaled(ncclGinMrcGPUContext* mrc, doca_gpu_dev_verbs_qp* qp, int peer,
                                                 doca_gpu_dev_verbs_addr raddr, doca_gpu_dev_verbs_addr laddr,
                                                 size_t bytes, const ncclGinMrcSignalImms& sig,
                                                 ncclGinMrcRqState* counterState, uint32_t counterId) {
  using nccl::utility::loadConst;
  const uint64_t num_data =
    (bytes == 0) ? 0 : doca_gpu_dev_verbs_div_ceil_aligned_pow2(bytes, DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE_SHIFT);
  uint32_t num_sink = (num_data > 0) ? (sig.n > 0 ? sig.n - 1 : 0) : sig.n;
  if (num_data == 0 && num_sink == 0) num_sink = 1; // counter-only op without data
  const uint64_t base_wqe_idx = doca_gpu_dev_verbs_reserve_wq_slots<resource_sharing_mode>(qp, num_data + num_sink);
  uint64_t wqe_idx = base_wqe_idx;

  size_t remaining = bytes;
  for (uint64_t i = 0; i < num_data; i++) {
    wqe_idx = base_wqe_idx + i;
    const size_t size_ =
      remaining > DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE ? DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE : remaining;
    const bool immHere = (i + 1 == num_data) && sig.n > 0;
    doca_gpu_dev_verbs_wqe_prepare_write(
      qp, doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx), wqe_idx,
      immHere ? DOCA_GPUNETIO_MLX5_OPCODE_RDMA_WRITE_IMM : DOCA_GPUNETIO_MLX5_OPCODE_RDMA_WRITE,
      DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE, immHere ? doca_gpu_dev_verbs_bswap32(sig.imm[0]) : 0,
      raddr.addr + (i * DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE), raddr.key,
      laddr.addr + (i * DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE), laddr.key, size_);
    remaining -= size_;
  }

  const uint64_t sinkRaddr = sizeof(uint64_t) * loadConst(&mrc->signals_sink_offset);
  const uint32_t sinkRkey = loadConst(loadConst(&mrc->signal_rkeys) + peer);
  const uint32_t sinkLkey = loadConst(&mrc->sink_lkey);
  const uint32_t firstSinkImm = (num_data > 0) ? 1 : 0;
  for (uint32_t k = 0; k < num_sink; k++) {
    wqe_idx = base_wqe_idx + num_data + k;
    const bool hasImm = firstSinkImm + k < sig.n;
    doca_gpu_dev_verbs_wqe_prepare_write(
      qp, doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx), wqe_idx,
      hasImm ? DOCA_GPUNETIO_MLX5_OPCODE_RDMA_WRITE_IMM : DOCA_GPUNETIO_MLX5_OPCODE_RDMA_WRITE,
      DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE, hasImm ? doca_gpu_dev_verbs_bswap32(sig.imm[firstSinkImm + k]) : 0,
      sinkRaddr, sinkRkey, 0, sinkLkey, sizeof(uint64_t));
  }

  doca_gpu_dev_verbs_mark_wqes_ready<resource_sharing_mode>(qp, base_wqe_idx, wqe_idx);
  if (counterState != nullptr) recordCounter(counterState, wqe_idx, counterId);
  doca_gpu_dev_verbs_submit<resource_sharing_mode, DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
                            DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB>(qp, wqe_idx + 1);
  return wqe_idx;
}

// Non-blocking send-CQ drain that advances counters for completed put-with-counter operations
// on one peer QP. Walks the SQ CQ from sq_cons_index while counter records are outstanding;
// each completed WQE that carries a recorded counterId bumps counters[counterId] by 1.
// Single-thread (shares the SQ CQ consumer with the wait path via GPU-scope atomics).
NCCL_DEVICE_INLINE static void sqDrainCounters(doca_gpu_dev_verbs_qp* qp, ncclGinMrcRqState* state,
                                               uint64_t* countersBuf) {
  if (state->counter_slots == nullptr || state->sq_status != 0) return;
  struct doca_gpu_dev_verbs_cq* cq = doca_gpu_dev_verbs_qp_get_cq_sq(qp);
  while (cuda::atomic_ref<unsigned int, cuda::thread_scope_device>(state->counter_pending)
           .load(cuda::memory_order_relaxed) != 0) {
    const uint64_t idx = state->sq_cons_index;
    const int status = doca_gpu_dev_verbs_poll_one_cq_at<
      DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU, DOCA_GPUNETIO_VERBS_QP_SQ>(cq, idx);
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

/* ── RDW get: server and response handling (run by the receive drain) ── */

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

// Serve one get request that arrived from `requester` on its QP: copy the descriptor out of the
// request ring, validate it, write each sub-request's data back, then answer with a
// Write-with-Immediate carrying the request id, generation and status. MRC places the earlier
// Writes on this QP before the response completes, so the requester sees the data once it
// consumes the response. Caller holds the requester's drain lock.
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode>
NCCL_DEVICE_INLINE static void rdwServe(ncclGinCtx ctx, ncclGinMrcGPUContext* mrc, doca_gpu_dev_verbs_qp* qp,
                                        int requester, uint32_t imm) {
  using nccl::utility::loadConst;
  const uint32_t window = loadConst(&mrc->rdw_window);
  const uint32_t slot = imm & kRdwIdMask & (window - 1);
  uint32_t* ring = (uint32_t*)(loadConst(&mrc->rdw_region) + rdwRingOffset(window, requester, slot));

  // The NIC wrote the slot; read it at system scope so no stale L1 copy from an earlier request
  // in the same slot is used.
  ncclGinMrcGetDesc desc;
  uint32_t* words = (uint32_t*)&desc;
  for (uint32_t i = 0; i < kRdwDescBytes / sizeof(uint32_t); i++)
    words[i] = doca_gpu_dev_verbs_load_relaxed_sys_global(ring + i);

  uint32_t status = kRdwStatusOk;
  uint32_t lkeys[kRdwMaxSubs];
  if (desc.tag != imm) {
    status = kRdwStatusRejected;
  } else if (desc.nsub == 0 || desc.nsub > kRdwMaxSubs) {
    status = kRdwStatusInvalid;
  } else {
    ncclGinMrcMrTable* table = loadConst(&mrc->mr_table);
    for (uint32_t s = 0; s < desc.nsub; s++) {
      const ncclGinMrcGetSub& sub = desc.sub[s];
      const uint64_t srcOff = ((uint64_t)sub.src_hi << 32) | sub.src_lo;
      if (sub.len == 0 || sub.len > kRdwMaxSubBytes ||
          !rdwLookupSource(table, sub.src_key, srcOff, sub.len, &lkeys[s])) {
        status = kRdwStatusInvalid;
        break;
      }
    }
  }

  if (status == kRdwStatusOk) {
    for (uint32_t s = 0; s < desc.nsub; s++) {
      const ncclGinMrcGetSub& sub = desc.sub[s];
      doca_gpu_dev_verbs_addr raddr, laddr;
      raddr.addr = ((uint64_t)sub.dst_hi << 32) | sub.dst_lo;
      raddr.key = sub.dst_key;
      laddr.addr = ((uint64_t)sub.src_hi << 32) | sub.src_lo;
      laddr.key = lkeys[s];
      write<resource_sharing_mode>(qp, raddr, laddr, sub.len, /*useImm=*/false, 0, nullptr, 0);
    }
  }

  // MRC rejects zero-length Write-with-Immediate: the response carries 8 scratch bytes into the
  // requester's response landing slot.
  doca_gpu_dev_verbs_addr raddr, laddr;
  raddr.addr = loadConst(&mrc->rdw_ctx_offset) + rdwRespOffset(ctx.nRanks, window);
  raddr.key = loadConst(loadConst(&mrc->rdw_rkeys) + requester);
  laddr.addr = 0;
  laddr.key = loadConst(&mrc->sink_lkey);
  write<resource_sharing_mode>(qp, raddr, laddr, kRdwRespBytes, /*useImm=*/true, rdwResponseImm(imm, status), nullptr,
                               0);
}

NCCL_DEVICE_INLINE static void rdwServeDispatch(ncclGinCtx ctx, ncclGinMrcGPUContext* mrc, doca_gpu_dev_verbs_qp* qp,
                                                int requester, uint32_t imm) {
  switch ((ncclGinResourceSharingMode)ctx.resourceSharingMode) {
  case NCCL_GIN_RESOURCE_SHARING_THREAD:
    rdwServe<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(ctx, mrc, qp, requester, imm);
    break;
  case NCCL_GIN_RESOURCE_SHARING_CTA:
    rdwServe<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA>(ctx, mrc, qp, requester, imm);
    break;
  default:
    rdwServe<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(ctx, mrc, qp, requester, imm);
    break;
  }
}

// Consume one get response from this peer. Responses of the current generation arrive in
// request order; anything from another generation is stale and dropped. A non-OK status or an
// out-of-order id is latched in get_status. Caller holds the peer's drain lock, so it is the
// only writer of get_completed.
NCCL_DEVICE_INLINE static void rdwComplete(ncclGinMrcGPUContext* mrc, ncclGinMrcRqState* state, uint32_t imm) {
  const uint32_t gen = (imm >> kRdwGenShift) & kRdwGenMask;
  if (gen != (nccl::utility::loadConst(&mrc->rdw_gen) & kRdwGenMask)) return;
  cuda::atomic_ref<uint64_t, cuda::thread_scope_device> completed(state->get_completed);
  const uint64_t done = completed.load(cuda::memory_order_relaxed);
  const uint32_t status = (imm >> kRdwStatusShift) & 3u;
  int32_t err = 0;
  if ((imm & kRdwIdMask) != ((uint32_t)done & kRdwIdMask))
    err = -EPROTO;
  else if (status != kRdwStatusOk)
    err = (int32_t)status;
  if (err != 0) {
    int32_t healthy = 0;
    cuda::atomic_ref<int32_t, cuda::thread_scope_device>(state->get_status)
      .compare_exchange_strong(healthy, err, cuda::memory_order_relaxed);
  }
  completed.store(done + 1, cuda::memory_order_release);
}

// Drain every peer's receive CQ once and apply the observed Write-with-Immediate signals to the
// local indexed signal buffer, then advance any completed put-with-counter operations by
// draining each peer's send CQ. Each Inc completion bumps signals[signalId] by 1; Add chunks are
// accumulated in the peer's drain state and the sum is applied on the last chunk (see
// encodeSignal). Each recorded, completed send WQE bumps counters[counterId] by 1. RDW get
// requests from a peer are served here, and responses to this rank's own get
// requests are consumed here. Non-blocking: intended to be called from the signal/counter wait
// loops so a waiting thread makes progress on its own context.
//
// Many CTAs share one GIN context, so this is called concurrently. rqPollOne/sqDrainCounters
// keep per-QP consumer indices that are NOT safe to advance from multiple threads at once (a
// race re-reads or regresses them, silently dropping signal increments), so each peer's drain
// runs under a per-peer try-lock: exactly one thread drains a given QP at a time. A thread that
// fails to acquire simply skips that peer this round; the lock holder drains every pending
// completion (including the signal that thread is waiting on), and every waiter keeps calling
// this from its spin loop, so no signal is missed and no thread starves.
NCCL_DEVICE_INLINE static void drainSignals(ncclGinCtx ctx) {
  using nccl::utility::loadConst;
  ncclGinMrcGPUContext* mrc = getCtx(ctx);
  ncclGinMrcRqState* rq = loadConst(&mrc->rq);
  if (rq == nullptr) return;
  uint64_t* signalBuf = loadConst(&mrc->signals);
  uint64_t* counterBuf = loadConst(&mrc->counters);
  for (int peer = 0; peer < ctx.nRanks; peer++) {
    doca_gpu_dev_verbs_qp* qp = getQp(mrc, peer);
    ncclGinMrcRqState* state = rq + peer;
    cuda::atomic_ref<int, cuda::thread_scope_device> drainLock(state->drain_lock);
    int unlocked = 0;
    if (!drainLock.compare_exchange_strong(unlocked, 1, cuda::memory_order_acquire, cuda::memory_order_relaxed))
      continue;
    uint32_t imm, opcode;
    while (rqPollOne(qp, state, &imm, &opcode) == 0) {
      if (opcode != kCqeRespWriteImm) continue;
      if (imm & kRdwFlag) {
        if (imm & kRdwResp)
          rdwComplete(mrc, state, imm);
        else
          rdwServeDispatch(ctx, mrc, qp, peer, imm);
      } else if (imm & kSigAddFlag) {
        // Chunks of one Add arrive back to back on this QP, and the drain lock makes this the
        // only thread touching add_acc/add_shift, even across drain rounds.
        state->add_acc += (uint64_t)(imm & kSigAddChunkMask) << state->add_shift;
        if (imm & kSigAddMore) {
          state->add_shift += kSigAddChunkBits;
        } else {
          const uint32_t signalId = (imm >> kSigAddChunkBits) & kSigAddIdMask;
          cuda::atomic_ref<uint64_t, cuda::thread_scope_system>(signalBuf[signalId])
            .fetch_add(state->add_acc, cuda::memory_order_release);
          state->add_acc = 0;
          state->add_shift = 0;
        }
      } else {
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system>(signalBuf[imm]).fetch_add(1, cuda::memory_order_release);
      }
    }
    sqDrainCounters(qp, state, counterBuf);
    drainLock.store(0, cuda::memory_order_release);
  }
}

/* ── Put / PutValue ───────────────────────────────────────────────── */

// Data-only puts use a plain RDMA Write; a put carrying a signal, or a bare signal, is delivered
// via Write-with-Immediate encoding the signal (see encodeSignal). A zero-byte put is sent like a
// bare signal (MRC rejects zero-length writes), and is a no-op when it carries neither a signal
// nor a counter.
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode, typename Coop>
NCCL_DEVICE_INLINE static void putImplMode(ncclGinCtx ctx, Coop coop, int peer, bool hasWins,
                                           ncclGinWindow_t dstWin, size_t dstOff, ncclGinWindow_t srcWin, size_t srcOff,
                                           size_t bytes, const ncclGinMrcSignalImms& sig, bool hasCounter,
                                           uint32_t counterId, cuda::thread_scope required, cuda::thread_scope given,
                                           uint32_t optFlags) {
  using nccl::utility::loadConst;
  (void)optFlags;
  coop.sync();
  const bool hasData = hasWins && bytes > 0;
  if (coop.thread_rank() == 0 && (hasData || sig.n > 0 || hasCounter)) {
    ncclGinMrcGPUContext* mrc = getCtx(ctx);
    doca_gpu_dev_verbs_qp* qp = getQp(mrc, peer);
    // Counter is advanced by the send-CQ drainer keyed off this peer's drain state.
    ncclGinMrcRqState* counterState = hasCounter ? (loadConst(&mrc->rq) + peer) : nullptr;

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
    const uint64_t lastWqeIdx = writeSignaled<resource_sharing_mode>(mrc, qp, peer, raddr, laddr,
                                                                     hasData ? bytes : 0, sig, counterState, counterId);
    if (!hasData) {
      // Confirm the local send completed before this thread (possibly the last one touching this
      // QP this round) moves on; bare signals otherwise have nothing else that consumes their SQ
      // CQE, so wait on this specific ticket rather than relying on a later reservation to
      // implicitly observe it.
      doca_gpu_dev_verbs_ticket_t sigTicket = static_cast<doca_gpu_dev_verbs_ticket_t>(lastWqeIdx);
      doca_gpu_dev_verbs_wait<resource_sharing_mode>(qp, &sigTicket);
    }
  }
  coop.sync();
}

template <typename Coop>
NCCL_DEVICE_INLINE static void putImpl(ncclGinCtx ctx, Coop coop, int peer, bool hasWins, ncclGinWindow_t dstWin,
                                       size_t dstOff, ncclGinWindow_t srcWin, size_t srcOff, size_t bytes,
                                       const ncclGinMrcSignalImms& sig, bool hasCounter, uint32_t counterId,
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
// 8-byte sink writes carry the signal immediates (MRC rejects zero-length writes). Same-QP
// ordering guarantees the value lands before the notification is observed.
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode, typename Coop, typename T>
NCCL_DEVICE_INLINE static void putValueImplMode(ncclGinCtx ctx, Coop coop, int peer, ncclGinWindow_t dstWin,
                                                size_t dstOff, T srcData, const ncclGinMrcSignalImms& sig,
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
    // The generic doca_gpu_dev_verbs_p switch instantiates a broken warp variant; this backend
    // always posts from a single thread, so call the thread variant directly.
    doca_gpu_dev_verbs_ticket_t ticket;
    doca_gpu_dev_verbs_p_thread<T, resource_sharing_mode>(qp, raddr, srcData, &ticket);
    if (sig.n > 0) {
      doca_gpu_dev_verbs_addr none = {};
      writeSignaled<resource_sharing_mode>(mrc, qp, peer, none, none, 0, sig, nullptr, 0);
    }
  }
  coop.sync();
}

template <typename Coop, typename T>
NCCL_DEVICE_INLINE static void putValueImpl(ncclGinCtx ctx, Coop coop, int peer, ncclGinWindow_t dstWin, size_t dstOff,
                                            T srcData, const ncclGinMrcSignalImms& sig, cuda::thread_scope required,
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
// `window` requests are in flight, and a full window is waited out by draining (which also
// serves the peer's own requests to this rank, so two ranks fetching from each other cannot
// starve one another).
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode>
NCCL_DEVICE_INLINE static void rdwPostGets(ncclGinCtx ctx, ncclGinMrcGPUContext* mrc, doca_gpu_dev_verbs_qp* qp,
                                           int peer, uint32_t srcKey, uint64_t srcOff, uint32_t dstKey,
                                           uint64_t dstOff, size_t bytes) {
  using nccl::utility::loadConst;
  ncclGinMrcRqState* state = loadConst(&mrc->rq) + peer;
  const uint32_t window = loadConst(&mrc->rdw_window);
  const uint32_t gen = loadConst(&mrc->rdw_gen);
  uint8_t* region = loadConst(&mrc->rdw_region);
  const uint64_t ctxOffset = loadConst(&mrc->rdw_ctx_offset);
  const uint32_t peerRkey = loadConst(loadConst(&mrc->rdw_rkeys) + peer);
  const uint32_t rdwLkey = loadConst(&mrc->rdw_lkey);

  cuda::atomic_ref<int, cuda::thread_scope_device> lock(state->get_lock);
  int unlocked = 0;
  while (!lock.compare_exchange_weak(unlocked, 1, cuda::memory_order_acquire, cuda::memory_order_relaxed))
    unlocked = 0;

  cuda::atomic_ref<uint64_t, cuda::thread_scope_device> completed(state->get_completed);
  cuda::atomic_ref<uint64_t, cuda::thread_scope_device> issued(state->get_issued);
  uint64_t id = issued.load(cuda::memory_order_relaxed);
  size_t done = 0;
  while (done < bytes) {
    NVCC_PRAGMA_UNROLL_DISABLED
    while (id - completed.load(cuda::memory_order_acquire) >= window) drainSignals(ctx);

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
    const uint32_t tag = rdwRequestImm(gen, id);
    desc->tag = tag;
    desc->nsub = nsub;
    // The NIC reads the descriptor from GPU memory once the doorbell rings.
    __threadfence_system();

    doca_gpu_dev_verbs_addr raddr, laddr;
    raddr.addr = ctxOffset + rdwRingOffset(window, ctx.rank, slot);
    raddr.key = peerRkey;
    laddr.addr = ctxOffset + rdwStagingOffset(ctx.nRanks, window, peer, slot);
    laddr.key = rdwLkey;
    write<resource_sharing_mode>(qp, raddr, laddr, kRdwDescBytes, /*useImm=*/true, tag, nullptr, 0);
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
      write<resource_sharing_mode>(qp, raddr, laddr, bytes, /*useImm=*/false, 0, nullptr, 0);
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
// answered, draining while spinning. Returns ncclRemoteError if any answer was not OK.
template <bool HasTimeout>
NCCL_DEVICE_INLINE static ncclResult_t rdwWaitGets(ncclGinCtx ctx, ncclGinMrcRqState* state, uint32_t ticket,
                                                   uint32_t* abortFlag, uint64_t startCycle, uint64_t timeoutCycles,
                                                   uint32_t& steps) {
  (void)startCycle;
  (void)timeoutCycles;
  cuda::atomic_ref<uint64_t, cuda::thread_scope_device> completed(state->get_completed);
  NVCC_PRAGMA_UNROLL_DISABLED
  while ((int32_t)((uint32_t)completed.load(cuda::memory_order_acquire) - ticket) < 0) {
    drainSignals(ctx);
    if NCCL_IF_CONSTEXPR (HasTimeout) {
      if (clock64() - startCycle >= timeoutCycles) return ncclTimeout;
    }
    if (nccl::utility::testAbort(abortFlag, steps)) return ncclSuccess;
  }
  return cuda::atomic_ref<int32_t, cuda::thread_scope_device>(state->get_status).load(cuda::memory_order_relaxed)
           ? ncclRemoteError
           : ncclSuccess;
}

// The void flush/wait variants have no way to return an error, and the fetched data cannot be
// trusted, so fail loudly.
NCCL_DEVICE_INLINE static void rdwFailGets(ncclGinCtx ctx) {
  printf("NCCL GIN/MRC: rank %d context %d: get failed (RDW error response); see ncclGinMrcRqState::get_status\n",
         ctx.rank, ctx.contextId);
  __trap();
}

/* ── Completion: flush / wait ─────────────────────────────────────── */

// The flush watermark is the current reservation index on the peer's QP, plus, for gets, the
// number of RDW requests issued to the peer: get data is only visible once their responses have
// been consumed.
template <enum doca_gpu_dev_verbs_resource_sharing_mode doca_sharing_mode>
NCCL_DEVICE_INLINE static void flushAsyncImpl(ncclGinCtx ctx, int peer, ncclGinRequest_t* outRequest) {
  ncclGinMrcRequest* req = reinterpret_cast<ncclGinMrcRequest*>(outRequest);
  ncclGinMrcGPUContext* mrc = getCtx(ctx);
  req->peer = peer;
  doca_gpu_dev_verbs_qp* qp = getQp(mrc, peer);
  req->sq_rsvd_index = doca_gpu_dev_verbs_atomic_read<uint64_t, doca_sharing_mode>(&qp->sq_rsvd_index);
  ncclGinMrcRqState* rq = nccl::utility::loadConst(&mrc->rq);
  req->get_ticket = rq ? (uint32_t)cuda::atomic_ref<uint64_t, cuda::thread_scope_device>(rq[peer].get_issued)
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
        doca_gpu_dev_verbs_ticket_t waitIdx = pollIdx;
        doca_gpu_dev_verbs_wait<doca_sharing_mode>(qp, &waitIdx);
        done = true;
      }
    }
    NVCC_PRAGMA_UNROLL_DISABLED
    while (!done) {
      int status = doca_gpu_dev_verbs_poll_one_cq_at<doca_sharing_mode>(doca_gpu_dev_verbs_qp_get_cq_sq(qp), pollIdx);
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

  ncclGinMrcRqState* rq = nccl::utility::loadConst(&mrc->rq);
  if (rq != nullptr && req.peer != ctx.rank) {
    ncclResult_t ret =
      rdwWaitGets<HasTimeout>(ctx, rq + req.peer, req.get_ticket, abortFlag, startCycle, timeoutCycles, steps);
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
  ncclGinMrcRqState* rq = loadConst(&mrc->rq);
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
        doca_gpu_dev_verbs_ticket_t waitIdx = ticket;
        doca_gpu_dev_verbs_wait<resource_sharing_mode>(&qps[peer], &waitIdx);
        done = true;
      }
    }

    NVCC_PRAGMA_UNROLL_DISABLED
    while (!done) {
      int status =
        doca_gpu_dev_verbs_poll_one_cq_at<resource_sharing_mode>(doca_gpu_dev_verbs_qp_get_cq_sq(&qps[peer]), ticket);
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

    // Gets from this peer are only locally visible once their RDW responses are consumed.
    // Loopback gets are plain Writes covered by the send-CQ wait above.
    if (rq != nullptr && peer != ctx.rank) {
      const uint32_t issued = (uint32_t)cuda::atomic_ref<uint64_t, cuda::thread_scope_device>(rq[peer].get_issued)
                                .load(cuda::memory_order_acquire);
      ncclResult_t ret = rdwWaitGets<HasTimeout>(ctx, rq + peer, issued, abortFlag, startCycle, timeoutCycles, steps);
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

} // namespace mrc
} // namespace gin
} // namespace nccl

template <>
struct ncclGinApi_Put<NCCL_NET_DEVICE_GIN_MRC> {
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, int peer, bool hasWins, ncclGinWindow_t dstWin,
                                      size_t dstOff, ncclGinWindow_t srcWin, size_t srcOff, size_t bytes,
                                      ncclGinSignalDescriptor signal, ncclGinSignalOp_t signalOp, uint64_t signalOpArg,
                                      bool hasCounter, ncclGinCounter_t counterId, bool hasDescriptor,
                                      ncclGinDescriptorSmem* descriptor, cuda::thread_scope required,
                                      cuda::thread_scope given, uint32_t optFlags = ncclGinOptFlagsDefault) {
    (void)hasDescriptor;
    (void)descriptor;
    // Signals ride Write-with-Immediate carrying the signal id (and Add value), so only INDEXED
    // signals are representable.
    assert(signal.type != NCCL_GIN_SIGNAL_TYPE_VA && "MRC GIN: VA signals are not supported");
    bool hasSignal = signal.type == NCCL_GIN_SIGNAL_TYPE_INDEXED;
    uint32_t signalId = hasSignal ? (uint32_t)signal.indexedSignal.signalId : 0;
    nccl::gin::mrc::putImpl(ctx, coop, peer, hasWins, dstWin, dstOff, srcWin, srcOff, bytes,
                            nccl::gin::mrc::encodeSignal(hasSignal, signalId, signalOp, signalOpArg), hasCounter,
                            (uint32_t)counterId, required, given, optFlags);
  }
};

template <>
struct ncclGinApi_PutValue<NCCL_NET_DEVICE_GIN_MRC> {
  template <typename Coop, typename T>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, int peer, ncclGinWindow_t dstWin, size_t dstOff,
                                      T srcVal, ncclGinSignalDescriptor signal, ncclGinSignalOp_t signalOp,
                                      uint64_t signalOpArg, bool hasDescriptor, ncclGinDescriptorSmem* descriptor,
                                      cuda::thread_scope required, cuda::thread_scope given,
                                      uint32_t optFlags = ncclGinOptFlagsDefault) {
    (void)hasDescriptor;
    (void)descriptor;
    assert(signal.type != NCCL_GIN_SIGNAL_TYPE_VA && "MRC GIN: VA signals are not supported");
    bool hasSignal = signal.type == NCCL_GIN_SIGNAL_TYPE_INDEXED;
    uint32_t signalId = hasSignal ? (uint32_t)signal.indexedSignal.signalId : 0;
    nccl::gin::mrc::putValueImpl(ctx, coop, peer, dstWin, dstOff, srcVal,
                                 nccl::gin::mrc::encodeSignal(hasSignal, signalId, signalOp, signalOpArg), required,
                                 given, optFlags);
  }
};

// MRC QPs have no remote-read verb: gets are served by the peer via Receiver-Driven Write.
template <>
struct ncclGinApi_Get<NCCL_NET_DEVICE_GIN_MRC> {
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, int peer, ncclGinWindow_t remoteWin, size_t remoteOff,
                                      ncclGinWindow_t localWin, size_t localOff, size_t bytes, bool hasDescriptor,
                                      ncclGinDescriptorSmem* descriptor, uint32_t optFlags = ncclGinOptFlagsDefault) {
    (void)hasDescriptor, (void)descriptor, (void)optFlags;
    nccl::gin::mrc::getImpl(ctx, coop, peer, remoteWin, remoteOff, localWin, localOff, bytes);
  }
};

template <>
struct ncclGinApi_FlushAsync<NCCL_NET_DEVICE_GIN_MRC> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, int peer, ncclGinRequest_t* outRequest, bool hasDescriptor,
                                      ncclGinDescriptorSmem* descriptor, uint32_t optFlags) {
    (void)hasDescriptor;
    (void)descriptor;
    (void)optFlags;
    switch ((ncclGinResourceSharingMode)ctx.resourceSharingMode) {
    case NCCL_GIN_RESOURCE_SHARING_THREAD:
      nccl::gin::mrc::flushAsyncImpl<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(ctx, peer, outRequest);
      break;
    case NCCL_GIN_RESOURCE_SHARING_CTA:
      nccl::gin::mrc::flushAsyncImpl<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA>(ctx, peer, outRequest);
      break;
    default:
      nccl::gin::mrc::flushAsyncImpl<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(ctx, peer, outRequest);
      break;
    }
  }
};

template <>
struct ncclGinApi_Wait<NCCL_NET_DEVICE_GIN_MRC> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, ncclGinRequest_t& request, bool hasDescriptor,
                                      ncclGinDescriptorSmem* descriptor, cuda::memory_order ord, uint32_t* abortFlag) {
    (void)hasDescriptor;
    (void)descriptor;
    switch ((ncclGinResourceSharingMode)ctx.resourceSharingMode) {
    case NCCL_GIN_RESOURCE_SHARING_THREAD:
      nccl::gin::mrc::waitImpl<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(ctx, request, ord, abortFlag);
      break;
    case NCCL_GIN_RESOURCE_SHARING_CTA:
      nccl::gin::mrc::waitImpl<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA>(ctx, request, ord, abortFlag);
      break;
    default:
      nccl::gin::mrc::waitImpl<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(ctx, request, ord, abortFlag);
      break;
    }
  }

  NCCL_DEVICE_INLINE static ncclResult_t call(ncclGinCtx ctx, ncclGinRequest_t& request, bool hasDescriptor,
                                              ncclGinDescriptorSmem* descriptor, cuda::memory_order ord,
                                              uint32_t* abortFlag, uint64_t timeoutCycles) {
    (void)hasDescriptor;
    (void)descriptor;
    switch ((ncclGinResourceSharingMode)ctx.resourceSharingMode) {
    case NCCL_GIN_RESOURCE_SHARING_THREAD:
      return nccl::gin::mrc::waitImpl<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(ctx, request, ord, abortFlag,
                                                                                           timeoutCycles);
    case NCCL_GIN_RESOURCE_SHARING_CTA:
      return nccl::gin::mrc::waitImpl<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA>(ctx, request, ord, abortFlag,
                                                                                     timeoutCycles);
    default:
      return nccl::gin::mrc::waitImpl<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(ctx, request, ord, abortFlag,
                                                                                     timeoutCycles);
    }
  }
};

template <>
struct ncclGinApi_Flush<NCCL_NET_DEVICE_GIN_MRC> {
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, bool hasDescriptor, ncclGinDescriptorSmem* descriptor,
                                      cuda::memory_order ord, uint32_t* abortFlag) {
    (void)hasDescriptor;
    (void)descriptor;
    nccl::gin::mrc::flushImpl(ctx, coop, ord, abortFlag);
  }

  template <typename Coop>
  NCCL_DEVICE_INLINE static ncclResult_t call(ncclGinCtx ctx, Coop coop, bool hasDescriptor,
                                              ncclGinDescriptorSmem* descriptor, cuda::memory_order ord,
                                              uint32_t* abortFlag, uint64_t timeoutCycles) {
    (void)hasDescriptor;
    (void)descriptor;
    return nccl::gin::mrc::flushImpl(ctx, coop, ord, abortFlag, timeoutCycles);
  }
};

// A signal is the immediate on the data write itself and travels the same QP, so the QP's
// ordered delivery already implies every earlier write on that QP has landed.
template <>
struct ncclGinApi_SupportsStrongSignal<NCCL_NET_DEVICE_GIN_MRC> {
  NCCL_DEVICE_INLINE static bool call(ncclGinCtx) {
    return true;
  }
};

template <>
struct ncclGinApi_FlushesAllPutsOnAnySignal<NCCL_NET_DEVICE_GIN_MRC> {
  // A signal is a Write-with-Immediate that only orders the puts issued before it on the same QP;
  // MRC has no NIC atomic that would flush puts received on other QPs, so the capability is absent.
  NCCL_DEVICE_INLINE static bool call(ncclGinCtx) {
    return false;
  }
};

template <>
struct ncclGinApi_BarrierOptions<NCCL_NET_DEVICE_GIN_MRC> {
  // Must match ncclGinProperties_t::barrierOptions reported by the MRC GIN plugin (v14 plugins
  // are mapped to the default barrier by NCCL).
  NCCL_DEVICE_INLINE static ncclGinBarrierOptions_t call(ncclGinCtx) {
    return NCCL_GIN_BARRIER_DEFAULT;
  }
};

template <>
struct ncclGinApi_ResetSignal<NCCL_NET_DEVICE_GIN_MRC> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, ncclGinSignalDescriptor signal) {
    assert(signal.type != NCCL_GIN_SIGNAL_TYPE_VA && "MRC GIN: VA signals are not supported");
    nccl::utility::loadConst(&nccl::gin::mrc::getCtx(ctx)->signals)[signal.indexedSignal.signalId] = 0;
  }
};

template <>
struct ncclGinApi_ResetCounter<NCCL_NET_DEVICE_GIN_MRC> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, ncclGinCounter_t counterId) {
    nccl::utility::loadConst(&nccl::gin::mrc::getCtx(ctx)->counters)[counterId] = 0;
  }
};

template <>
struct ncclGinApi_GetSignalPtr<NCCL_NET_DEVICE_GIN_MRC> {
  NCCL_DEVICE_INLINE static ncclGinOffsetPtr call(ncclGinCtx ctx, ncclGinSignal_t signalId) {
    return {nccl::utility::loadConst(&nccl::gin::mrc::getCtx(ctx)->signals) + signalId, 0};
  }
};

template <>
struct ncclGinApi_GetCounterPtr<NCCL_NET_DEVICE_GIN_MRC> {
  NCCL_DEVICE_INLINE static ncclGinOffsetPtr call(ncclGinCtx ctx, ncclGinCounter_t counterId) {
    return {nccl::utility::loadConst(&nccl::gin::mrc::getCtx(ctx)->counters) + counterId, 0};
  }
};

// Signals and counters only become locally visible when their CQEs are drained, so the wait
// loops must call this or they spin forever.
template <>
struct ncclGinApi_Progress<NCCL_NET_DEVICE_GIN_MRC> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx) {
    nccl::gin::mrc::drainSignals(ctx);
  }
};

#endif /* _NCCL_DEVICE_GIN_MRC_H_ */
