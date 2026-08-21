/*************************************************************************
 * SPDX-License-Identifier: Apache-2.0
 *
 * MRC GIN device handle structs, shared between NCCL device code and the
 * out-of-tree host plugin (libnccl-gin-mrc.so), which populates them in GPU
 * memory from createContext().
 *
 * IMPORTANT: Must stay in sync with the plugin-side definition. Bump
 * mrcBackendMinVersions[] in src/gin/gin_host.cc on any layout change.
 *
 * Deliberately free of DOCA/CUDA/verbs dependencies so the plugin can include
 * it next to the real DOCA headers without type clashes; opaque pointers are
 * cast to their DOCA types in gin_mrc.h.
 *************************************************************************/

#ifndef _NCCL_DEVICE_GIN_MRC_DEV_H_
#define _NCCL_DEVICE_GIN_MRC_DEV_H_

#include <stdint.h>

// Per-peer drain state. MRC QPs have no atomics, so a peer's signal arrives as an RDMA
// Write-with-Immediate and the receiver drains its receive CQ to observe it;
// consumer_index is the running count of consumed receive CQEs.
//
// Counters likewise have no atomic path, so they are advanced on the send side by draining
// the local send CQ: a put-with-counter records counterId+1 in counter_slots keyed by its
// last WQE index, and the SQ-CQ drainer bumps counters[counterId] once that WQE completes.
struct ncclGinMrcRqState {
  uint64_t consumer_index;
  uint32_t rq_depth;
  int32_t status;
  // Send-CQ counter drain state.
  uint64_t sq_cons_index;       // SQ-CQ walk position for counter draining
  uint32_t* counter_slots;      // [sq_wqe_mask+1] ring of counterId+1 (0 = no counter)
  uint32_t sq_wqe_mask;         // counter_slots capacity minus one (power of two)
  int32_t sq_status;            // latched SQ-CQ error (0 = healthy)
  unsigned int counter_pending; // outstanding recorded counter ops (drain gate)
  // RQ/SQ drain must be single-threaded per QP. Many CTAs share one GIN context, so this
  // device-scope lock (0=free, 1=held) serializes the per-peer drain: racing on
  // consumer_index regresses it and silently drops signal increments.
  int drain_lock;
};

struct ncclGinMrcGPUContext {
  void* qps;                   // doca_gpu_dev_verbs_qp[nRanks], one per peer
  struct ncclGinMrcRqState* rq; // [nRanks]

  uint64_t* signals;   // local indexed signal counters, maintained by the receive drain
  uint32_t* signal_rkeys; // [nRanks] big-endian rkeys of the peers' signal MRs
  uint64_t* counters;  // local indexed counters, maintained by the send-CQ drain

  // Element index, within the signal MR, of a scratch slot reserved for the payload of
  // bare-signal Write-with-Immediate ops. MRC rejects zero-length Write-with-Immediate, so
  // those ops must land their bytes somewhere; targeting the signal slot itself would
  // overwrite the count the receiver's drain maintains there.
  uint32_t signals_sink_offset;

  uint32_t sink_lkey; // big-endian lkey of a local scratch buffer we write from but never read
};

struct ncclGinMrcMemHandle {
  uint32_t* rkeys; // [nRanks], big-endian
  uint32_t lkey;   // big-endian
};

#endif /* _NCCL_DEVICE_GIN_MRC_DEV_H_ */
