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
 * cast to their DOCA types in the device backend.
 *************************************************************************/

#ifndef _NCCL_DEVICE_GIN_MRC_DEV_H_
#define _NCCL_DEVICE_GIN_MRC_DEV_H_

#include <stdint.h>

// MRC QPs have no atomics: a signal is the immediate of an RDMA Write-with-Immediate, so every QP
// has a receive queue. The receive queues are managed by the CPU: device code from the open-source
// GPUNetIO NCCL bundles posts with doca_gpu_dev_verbs_put_immediate, and the plugin drains the
// receive queues into per-context signal maps, with the semantics of the open-source
// doca_gpu_verbs_rq_progress.
// NCCL passes this version to the plugin's createContext as the backend version; it is the layout
// of the structs below that the device code reads. History: 7/9 were the CPU RQ layouts of the
// combined CPU/GPU RQ code (8/10 its GPU RQ layouts, 12 the GPU RQ branch); 11 drops the GPU RQ
// fields.
#define NCCL_GIN_MRC_VERSION 11

// Signal ids of a context. The signal map adds each immediate's value to the GPU slot
// registered for its id. Each context owns one signal table, indexed by id:
//   [0, nSignals)                 NCCL indexed signals
//   rdw_signal_base + p           RDW get requests received from rank p
//   rdw_signal_base + nRanks + p  RDW get responses from rank p (OK)
//   rdw_signal_base + 2*nRanks+p  RDW get responses from rank p (error)
// with rdw_signal_base = nSignals. Every rank uses the same layout, so a sender addresses the
// receiver's slot by id alone.
// The NIC writes no slot (a bare signal is a zero-length Write-with-Immediate), so the table has no
// MR: it is written only by the signal map.
#define NCCL_GIN_MRC_SIGNAL_TABLE_SIZE(nSignals, nRanks) ((nSignals) + 3 * (nRanks))

// Per-peer device state.
//
// Counters have no atomic path either, so they are advanced on the send side by draining the
// local send CQ: a put-with-counter records counterId+1 in counter_slots keyed by its last WQE
// index, and the SQ-CQ drainer bumps counters[counterId] once that WQE completes.
struct ncclGinMrcPeerState {
  // Send-CQ counter drain state.
  uint64_t sq_cons_index;       // SQ-CQ walk position for counter draining
  uint32_t* counter_slots;      // [sq_wqe_mask+1] ring of counterId+1 (0 = no counter)
  uint32_t sq_wqe_mask;         // counter_slots capacity minus one (power of two)
  int32_t sq_status;            // latched SQ-CQ error (0 = healthy)
  unsigned int counter_pending; // outstanding recorded counter ops (drain gate)
  // Serializes the progress work on this peer (receive drain, counter drain, RDW serving): many
  // CTAs share one GIN context, and the drains walk per-peer positions that must advance from one
  // thread at a time.
  int drain_lock; // 0=free, 1=held
  // RDW get, requester side: requests posted to this peer. Ids are allocated under get_lock in
  // wire order.
  uint64_t get_issued;
  int get_lock; // 0=free, 1=held

  // RDW get, server side: requests from this peer served so far. Requests are served in order, and
  // the peer's request counter (signal rdw_signal_base + peer) says how many arrived.
  uint64_t rdw_served;
};

// Registered windows of this rank, used by the RDW server to validate a get request's source
// key and bounds and to translate the requester-supplied rkey into the local lkey.
#define NCCL_GIN_MRC_MR_TABLE_CAPACITY 1024
// RDW get wire constants shared with the host plugin (which sizes the RDW buffer).
#define NCCL_GIN_MRC_RDW_DESC_BYTES 128 // request descriptor and request-ring slot size
struct ncclGinMrcMrEntry {
  uint32_t rkey; // big-endian
  uint32_t lkey; // big-endian
  uint64_t size; // registered bytes (MRs are zero-based, so offsets must stay below this)
};
struct ncclGinMrcMrTable {
  uint32_t count;
  uint32_t reserved;
  struct ncclGinMrcMrEntry entries[NCCL_GIN_MRC_MR_TABLE_CAPACITY];
};

struct ncclGinMrcGPUContext {
  // The QP to each peer: an array of the open-source GPUNetIO doca_gpu_dev_verbs_qp, one per peer.
  void* qps;
  struct ncclGinMrcPeerState* peers; // [nRanks]

  // This context's signal table (id layout above), indexed by signal id.
  uint64_t* signals;
  // [nSignals] reset baselines. The host signal map is the only writer of signal slots and adds to
  // a host-side shadow of each, so a device-side store would be overwritten; a reset records the
  // slot here instead, and the logical value of signal i is signals[i] - signal_base[i].
  uint64_t* signal_base;
  uint32_t rdw_signal_base; // see the id layout above
  // Nonzero when the plugin was asked not to build the self loop (NCCL_GIN_MRC_SELF_LOOP=0): the QP
  // to the local rank is left unconnected and there is no loopback responder, so a put, signal or
  // get to the local rank traps instead of hanging.
  uint32_t self_loop_off;

  uint64_t* counters; // local indexed counters, maintained by the send-CQ drain

  // Receiver-Driven Write (RDW) get. MRC has no RDMA Read, so a get is a request descriptor
  // sent by Write-with-Immediate into the data owner's request ring; the owner's device progress
  // serves it with RDMA Writes plus a Write-with-Immediate response on the same QP.
  //
  // RDW MR layout, one region per context (identical on every rank):
  //   [0, R)        request ring  [nRanks requester][rdw_window] x 128 B (written by peers)
  //   [R, 2R)       staging       [nRanks peer][rdw_window] x 128 B (local descriptors)
  // where R = nRanks * rdw_window * 128. A response is a zero-length Write-with-Immediate.
  struct ncclGinMrcMrTable* mr_table; // this rank's registered windows
  uint8_t* rdw_region;     // device pointer to this context's region of the local RDW MR
  uint32_t* rdw_rkeys;     // [nRanks] big-endian rkeys of the peers' RDW MRs
  uint64_t rdw_ctx_offset; // offset of this context's region within the (zero-based) RDW MR
  uint32_t rdw_lkey;       // big-endian
  uint32_t rdw_window;     // in-flight requests per peer (power of two)
};

struct ncclGinMrcMemHandle {
  uint32_t* rkeys; // [nRanks], big-endian
  uint32_t lkey;   // big-endian
};

#endif /* _NCCL_DEVICE_GIN_MRC_DEV_H_ */
