/*************************************************************************
 * SPDX-License-Identifier: Apache-2.0
 *
 * MRC implementations of the NCCL GIN device-side API.
 *
 * MRC QPs speak RDMA Write and RDMA Write-with-Immediate only: no atomics, no
 * remote read. So a signal is the immediate of a Write-with-Immediate, which
 * consumes a receive WQE of the target; a counter is advanced locally by
 * draining the send CQ; only INDEXED signals are representable; and Get is a
 * Receiver-Driven Write (RDW) served by the data owner's device code, only
 * while it runs a kernel that drives GIN progress (any signal/counter read or
 * wait, flush, or wait on a request).
 *
 * The receive queues are managed by the CPU: the host plugin drains them into
 * per-context signal maps, and device code posts with the open-source DOCA
 * GPUNetIO doca_gpu_dev_verbs_put_immediate (gin_mrc_cpu_rq.h).
 *
 * The GPU-resident context is created by the out-of-tree host plugin; its layout
 * is gin_mrc_dev.h.
 *************************************************************************/

#ifndef _NCCL_DEVICE_GIN_MRC_H_
#define _NCCL_DEVICE_GIN_MRC_H_

#include <errno.h>
#include <cassert>
#include <cstdint>

#if defined(NCCL_GIN_MRC_GPU_RQ) && NCCL_GIN_MRC_GPU_RQ
#error "This NCCL implements MRC GIN with CPU-managed receive queues only (NCCL_GIN_MRC_GPU_RQ is not supported)"
#endif

#include "nccl_device/utility.h"
#include "../gin_device_common.h"
#include "gin_mrc_dev.h"

struct ncclGinMrcRequest {
  int peer;
  uint32_t get_ticket; // low 32 bits of the peer's get_issued at flushAsync time
  uint64_t sq_rsvd_index;
};
static_assert(sizeof(ncclGinMrcRequest) <= sizeof(ncclGinRequest_t),
              "ncclGinMrcRequest must fit in ncclGinRequest_t");

// gin_mrc_cpu_rq.h implements the functions in nccl::gin::mrc for the API below: getCtx,
// encodeSignal, putImpl, putValueImpl, getImpl, flushAsyncImpl, waitImpl, flushImpl, progress,
// resetSignal and signalPtr.
#include "gin_mrc_cpu_rq.h"

namespace nccl {
namespace gin {
namespace mrc {
// Without the self loop (NCCL_GIN_MRC_SELF_LOOP=0) the QP to the local rank is not connected, so
// an operation on it would never complete. Self traffic should go over LSA instead; GIN barriers
// fenced with ncclGinFenceLevel::Put (the default) signal the local rank too.
NCCL_DEVICE_INLINE static void checkSelfLoop(ncclGinCtx ctx, int peer) {
  if (peer == ctx.rank && nccl::utility::loadConst(&getCtx(ctx)->self_loop_off)) {
    printf("NCCL GIN/MRC: rank %d context %d: GIN put/signal/get to the local rank needs the self loop "
           "(NCCL_GIN_MRC_SELF_LOOP=1)\n",
           ctx.rank, ctx.contextId);
    __trap();
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
    nccl::gin::mrc::checkSelfLoop(ctx, peer);
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
    nccl::gin::mrc::checkSelfLoop(ctx, peer);
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
    nccl::gin::mrc::checkSelfLoop(ctx, peer);
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
    nccl::gin::mrc::resetSignal(ctx, (uint32_t)signal.indexedSignal.signalId);
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
    return nccl::gin::mrc::signalPtr(ctx, (uint32_t)signalId);
  }
};

template <>
struct ncclGinApi_GetCounterPtr<NCCL_NET_DEVICE_GIN_MRC> {
  NCCL_DEVICE_INLINE static ncclGinOffsetPtr call(ncclGinCtx ctx, ncclGinCounter_t counterId) {
    return {nccl::utility::loadConst(&nccl::gin::mrc::getCtx(ctx)->counters) + counterId, 0};
  }
};

// Device-side progress (counters and RDW get serving): the wait loops must call this or they spin
// forever.
template <>
struct ncclGinApi_Progress<NCCL_NET_DEVICE_GIN_MRC> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx) {
    nccl::gin::mrc::progress(ctx);
  }
};

#endif /* _NCCL_DEVICE_GIN_MRC_H_ */
