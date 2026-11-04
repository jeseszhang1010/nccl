/*************************************************************************
 * SPDX-License-Identifier: Apache-2.0
 *
 * MRC implementations of the NCCL GIN device-side API.
 *
 * MRC QPs speak RDMA Write and RDMA Write-with-Immediate only: no atomics, no
 * remote read. Consequences that shape this backend:
 *   - A signal is an inbound Write-with-Immediate carrying the signal id. The
 *     receiver posts receive WQEs and drains its receive CQ to apply them to a
 *     local signal array (see drainSignals / ncclGinApi_Progress).
 *   - A counter is advanced locally by draining the send CQ: a put-with-counter
 *     records its counter id keyed by the last WQE index, and the drain bumps
 *     the counter once that WQE completes.
 *   - Only INDEXED signals are representable; VA signals and Get are unsupported.
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
// Write-with-Immediate carrying signalId (bytes == 0 posts a single zero-length
// Write-with-Immediate, i.e. a bare signal). When counterState is non-null the last WQE index
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

// Drain every peer's receive CQ once and apply the observed Write-with-Immediate signals to the
// local indexed signal buffer, then advance any completed put-with-counter operations by
// draining each peer's send CQ. Each receive completion bumps signals[signalId] by 1 (MRC
// signals carry Inc semantics); each recorded, completed send WQE bumps counters[counterId] by
// 1. Non-blocking: intended to be called from the signal/counter wait loops so a waiting thread
// makes progress on its own context.
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
      if (opcode == kCqeRespWriteImm)
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system>(signalBuf[imm]).fetch_add(1, cuda::memory_order_release);
    }
    sqDrainCounters(qp, state, counterBuf);
    drainLock.store(0, cuda::memory_order_release);
  }
}

/* ── Put / PutValue ───────────────────────────────────────────────── */

// Data-only puts use a plain RDMA Write; a put carrying a signal, or a bare signal, is
// delivered via Write-with-Immediate encoding the signal id.
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode, typename Coop>
NCCL_DEVICE_INLINE static void putImplMode(ncclGinCtx ctx, Coop coop, int peer, bool hasWins,
                                           ncclGinWindow_t dstWin, size_t dstOff, ncclGinWindow_t srcWin, size_t srcOff,
                                           size_t bytes, bool hasSignal, uint32_t signalId, bool hasCounter,
                                           uint32_t counterId, cuda::thread_scope required, cuda::thread_scope given,
                                           uint32_t optFlags) {
  using nccl::utility::loadConst;
  (void)optFlags;
  coop.sync();
  if (coop.thread_rank() == 0) {
    ncclGinMrcGPUContext* mrc = getCtx(ctx);
    doca_gpu_dev_verbs_qp* qp = getQp(mrc, peer);
    ncclGinMrcMemHandle* dstMh = (ncclGinMrcMemHandle*)dstWin;
    ncclGinMrcMemHandle* srcMh = (ncclGinMrcMemHandle*)srcWin;
    // Counter is advanced by the send-CQ drainer keyed off this peer's drain state.
    ncclGinMrcRqState* counterState = hasCounter ? (loadConst(&mrc->rq) + peer) : nullptr;

    if ((required == cuda::thread_scope_system) && (given > required)) {
      doca_gpu_dev_verbs_fence_release<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
    }

    if (hasWins) {
      doca_gpu_dev_verbs_addr raddr, laddr;
      raddr.addr = dstOff;
      raddr.key = loadConst(loadConst(&dstMh->rkeys) + peer);
      laddr.addr = srcOff;
      laddr.key = loadConst(&srcMh->lkey);
      // Data write: carry the signal as an immediate when requested, otherwise a plain write.
      write<resource_sharing_mode>(qp, raddr, laddr, bytes, /*useImm=*/hasSignal, signalId, counterState, counterId);
    } else if (hasSignal) {
      // Bare signal: the immediate carries the signalId; the receiver's drain increments its
      // LOCAL signal counter from the CQE. Transfer the registered sink scratch because MRC does
      // not accept a zero-length Write-with-Immediate. The payload must land in the reserved sink
      // slot, never in the signal slot, which the receiver's drain owns.
      doca_gpu_dev_verbs_addr sig_raddr, sig_laddr;
      sig_raddr.addr = sizeof(uint64_t) * loadConst(&mrc->signals_sink_offset);
      sig_raddr.key = loadConst(loadConst(&mrc->signal_rkeys) + peer);
      sig_laddr.addr = 0;
      sig_laddr.key = loadConst(&mrc->sink_lkey);
      uint64_t sigWqeIdx = write<resource_sharing_mode>(qp, sig_raddr, sig_laddr, sizeof(uint64_t), /*useImm=*/true,
                                                        signalId, counterState, counterId);
      // Confirm the local send completed before this thread (possibly the last one touching this
      // QP this round) moves on; bare signals otherwise have nothing else that consumes their SQ
      // CQE, so wait on this specific ticket rather than relying on a later reservation to
      // implicitly observe it.
      doca_gpu_dev_verbs_ticket_t sigTicket = static_cast<doca_gpu_dev_verbs_ticket_t>(sigWqeIdx);
      doca_gpu_dev_verbs_wait<resource_sharing_mode>(qp, &sigTicket);
    }
  }
  coop.sync();
}

template <typename Coop>
NCCL_DEVICE_INLINE static void putImpl(ncclGinCtx ctx, Coop coop, int peer, bool hasWins, ncclGinWindow_t dstWin,
                                       size_t dstOff, ncclGinWindow_t srcWin, size_t srcOff, size_t bytes,
                                       bool hasSignal, uint32_t signalId, bool hasCounter, uint32_t counterId,
                                       cuda::thread_scope required, cuda::thread_scope given, uint32_t optFlags) {
  switch ((ncclGinResourceSharingMode)ctx.resourceSharingMode) {
  case NCCL_GIN_RESOURCE_SHARING_THREAD:
    putImplMode<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(ctx, coop, peer, hasWins, dstWin, dstOff, srcWin,
                                                                     srcOff, bytes, hasSignal, signalId, hasCounter,
                                                                     counterId, required, given, optFlags);
    break;
  case NCCL_GIN_RESOURCE_SHARING_CTA:
    putImplMode<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA>(ctx, coop, peer, hasWins, dstWin, dstOff, srcWin, srcOff,
                                                               bytes, hasSignal, signalId, hasCounter, counterId,
                                                               required, given, optFlags);
    break;
  default:
    putImplMode<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(ctx, coop, peer, hasWins, dstWin, dstOff, srcWin, srcOff,
                                                               bytes, hasSignal, signalId, hasCounter, counterId,
                                                               required, given, optFlags);
    break;
  }
}

// The small value is delivered by an inline RDMA Write; when a signal is requested, a following
// 8-byte Write-with-Immediate carries the signal id (the fabric rejects zero-length
// Write-with-Immediate). Same-QP ordering guarantees the value lands before the notification is
// observed.
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode, typename Coop, typename T>
NCCL_DEVICE_INLINE static void putValueImplMode(ncclGinCtx ctx, Coop coop, int peer, ncclGinWindow_t dstWin,
                                                size_t dstOff, T srcData, bool hasSignal, uint32_t signalId,
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
    if (hasSignal) {
      // The fabric rejects a zero-length Write-with-Immediate; transfer the registered 8-byte
      // sink scratch into the reserved sink slot instead (content unused, only the immediate
      // value matters; the signal slot itself is owned by the receiver's drain).
      doca_gpu_dev_verbs_addr sig_raddr, sig_laddr;
      sig_raddr.addr = sizeof(uint64_t) * loadConst(&mrc->signals_sink_offset);
      sig_raddr.key = loadConst(loadConst(&mrc->signal_rkeys) + peer);
      sig_laddr.addr = 0;
      sig_laddr.key = loadConst(&mrc->sink_lkey);
      write<resource_sharing_mode>(qp, sig_raddr, sig_laddr, sizeof(uint64_t), /*useImm=*/true, signalId, nullptr, 0);
    }
  }
  coop.sync();
}

template <typename Coop, typename T>
NCCL_DEVICE_INLINE static void putValueImpl(ncclGinCtx ctx, Coop coop, int peer, ncclGinWindow_t dstWin, size_t dstOff,
                                            T srcData, bool hasSignal, uint32_t signalId, cuda::thread_scope required,
                                            cuda::thread_scope given, uint32_t optFlags) {
  switch ((ncclGinResourceSharingMode)ctx.resourceSharingMode) {
  case NCCL_GIN_RESOURCE_SHARING_THREAD:
    putValueImplMode<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(ctx, coop, peer, dstWin, dstOff, srcData,
                                                                          hasSignal, signalId, required, given,
                                                                          optFlags);
    break;
  case NCCL_GIN_RESOURCE_SHARING_CTA:
    putValueImplMode<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA>(ctx, coop, peer, dstWin, dstOff, srcData, hasSignal,
                                                                    signalId, required, given, optFlags);
    break;
  default:
    putValueImplMode<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(ctx, coop, peer, dstWin, dstOff, srcData, hasSignal,
                                                                    signalId, required, given, optFlags);
    break;
  }
}

/* ── Completion: flush / wait ─────────────────────────────────────── */

// MRC has no Get, so there is never read data to settle: the flush watermark is simply the
// current reservation index on the peer's QP.
template <enum doca_gpu_dev_verbs_resource_sharing_mode doca_sharing_mode>
NCCL_DEVICE_INLINE static void flushAsyncImpl(ncclGinCtx ctx, int peer, ncclGinRequest_t* outRequest) {
  ncclGinMrcRequest* req = reinterpret_cast<ncclGinMrcRequest*>(outRequest);
  req->peer = peer;
  doca_gpu_dev_verbs_qp* qp = getQp(getCtx(ctx), peer);
  req->sq_rsvd_index = doca_gpu_dev_verbs_atomic_read<uint64_t, doca_sharing_mode>(&qp->sq_rsvd_index);
}

template <bool HasTimeout, enum doca_gpu_dev_verbs_resource_sharing_mode doca_sharing_mode>
NCCL_DEVICE_INLINE static ncclResult_t waitImplCore(ncclGinCtx ctx, ncclGinRequest_t& request, cuda::memory_order ord,
                                                    uint32_t* abortFlag, uint64_t timeoutCycles) {
  using nccl::utility::testAbort;
  (void)timeoutCycles; // referenced only when HasTimeout is true
  ncclGinMrcRequest& req = reinterpret_cast<ncclGinMrcRequest&>(request);
  doca_gpu_dev_verbs_qp* qp = getQp(getCtx(ctx), req.peer);
  if (req.sq_rsvd_index == 0) return ncclSuccess;
  const doca_gpu_dev_verbs_ticket_t pollIdx = static_cast<doca_gpu_dev_verbs_ticket_t>(req.sq_rsvd_index - 1u);

  // doca_gpu_dev_verbs_wait blocks indefinitely, so it can only be used when the caller has no
  // deadline AND no abort flag to honor.
  if NCCL_IF_CONSTEXPR (!HasTimeout) {
    if (abortFlag == nullptr) {
      doca_gpu_dev_verbs_ticket_t waitIdx = pollIdx;
      doca_gpu_dev_verbs_wait<doca_sharing_mode>(qp, &waitIdx);
      cuda::atomic_thread_fence(ord, cuda::thread_scope_system);
      return ncclSuccess;
    }
  }

  uint32_t steps = 0;
  uint64_t startCycle = 0;
  (void)startCycle; // referenced only when HasTimeout is true
  if NCCL_IF_CONSTEXPR (HasTimeout) startCycle = clock64();
  NVCC_PRAGMA_UNROLL_DISABLED
  while (true) {
    int status = doca_gpu_dev_verbs_poll_one_cq_at<doca_sharing_mode>(doca_gpu_dev_verbs_qp_get_cq_sq(qp), pollIdx);
    if (status == 0) {
      cuda::atomic_thread_fence(ord, cuda::thread_scope_system);
      return ncclSuccess;
    }
    if NCCL_IF_CONSTEXPR (HasTimeout) {
      if (clock64() - startCycle >= timeoutCycles) return ncclTimeout;
    }
    if (testAbort(abortFlag, steps)) {
      cuda::atomic_thread_fence(ord, cuda::thread_scope_system);
      return ncclSuccess;
    }
  }
}

template <enum doca_gpu_dev_verbs_resource_sharing_mode doca_sharing_mode>
NCCL_DEVICE_INLINE static void waitImpl(ncclGinCtx ctx, ncclGinRequest_t& request, cuda::memory_order ord,
                                        uint32_t* abortFlag) {
  (void)waitImplCore</*HasTimeout=*/false, doca_sharing_mode>(ctx, request, ord, abortFlag, 0);
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
  uint32_t steps = 0;
  uint64_t startCycle = 0;
  (void)startCycle; // referenced only when HasTimeout is true
  if NCCL_IF_CONSTEXPR (HasTimeout) startCycle = clock64();

  NVCC_PRAGMA_UNROLL_DISABLED
  for (int peer = coop.thread_rank(); peer < ctx.nRanks; peer += coop.size()) {
    uint64_t ticket = doca_gpu_dev_verbs_atomic_read<uint64_t, resource_sharing_mode>(&qps[peer].sq_rsvd_index);
    if (ticket == 0) continue; // No need to poll if there are no outstanding operations
    --ticket;

    if NCCL_IF_CONSTEXPR (!HasTimeout) {
      if (abortFlag == nullptr) {
        doca_gpu_dev_verbs_ticket_t waitIdx = ticket;
        doca_gpu_dev_verbs_wait<resource_sharing_mode>(&qps[peer], &waitIdx);
        continue;
      }
    }

    NVCC_PRAGMA_UNROLL_DISABLED
    while (true) {
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
    (void)signalOp;
    (void)signalOpArg;
    (void)hasDescriptor;
    (void)descriptor;
    // Signals ride a Write-with-Immediate carrying the signal id, so only INDEXED signals are
    // representable and each delivery is an Inc.
    assert(signal.type != NCCL_GIN_SIGNAL_TYPE_VA && "MRC GIN: VA signals are not supported");
    bool hasSignal = signal.type == NCCL_GIN_SIGNAL_TYPE_INDEXED;
    uint32_t signalId = hasSignal ? (uint32_t)signal.indexedSignal.signalId : 0;
    nccl::gin::mrc::putImpl(ctx, coop, peer, hasWins, dstWin, dstOff, srcWin, srcOff, bytes, hasSignal, signalId,
                            hasCounter, (uint32_t)counterId, required, given, optFlags);
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
    (void)signalOp;
    (void)signalOpArg;
    (void)hasDescriptor;
    (void)descriptor;
    assert(signal.type != NCCL_GIN_SIGNAL_TYPE_VA && "MRC GIN: VA signals are not supported");
    bool hasSignal = signal.type == NCCL_GIN_SIGNAL_TYPE_INDEXED;
    uint32_t signalId = hasSignal ? (uint32_t)signal.indexedSignal.signalId : 0;
    nccl::gin::mrc::putValueImpl(ctx, coop, peer, dstWin, dstOff, srcVal, hasSignal, signalId, required, given,
                                 optFlags);
  }
};

// MRC QPs have no remote-read verb.
template <>
struct ncclGinApi_Get<NCCL_NET_DEVICE_GIN_MRC> {
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, int peer, ncclGinWindow_t remoteWin, size_t remoteOff,
                                      ncclGinWindow_t localWin, size_t localOff, size_t bytes, bool hasDescriptor,
                                      ncclGinDescriptorSmem* descriptor, uint32_t optFlags = ncclGinOptFlagsDefault) {
    (void)ctx, (void)coop, (void)peer, (void)remoteWin, (void)remoteOff, (void)localWin, (void)localOff, (void)bytes,
      (void)hasDescriptor, (void)descriptor, (void)optFlags;
    assert(false && "MRC GIN: Get is not supported");
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
