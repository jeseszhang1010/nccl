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
// has a receive queue. The receive queues are managed by the GPU: the plugin exports its QPs to
// DOCA SDK GPU handlers (doca_gpu_verbs_export_qp), and device code built against the DOCA SDK
// device headers posts the receive WQEs and drains the receive CQs itself.
// NCCL passes this version to the plugin's createContext as the backend version; it is the layout
// of the structs below that the device code reads. History: 8/10 were the GPU RQ layouts of the
// combined CPU/GPU RQ code (7/9 its CPU RQ layouts, 11 the CPU RQ branch); 12 drops the CPU RQ
// fields.
#define NCCL_GIN_MRC_VERSION 12

// How receive WQEs are posted/reposted in GPU RQ mode (ncclGinMrcPeerState::post_mode). Selected by
// the host plugin at runtime via NCCL_GIN_MRC_RQ_POST_MODE.
//   DBR:  lightweight. The ring is filled once with immutable notification WQEs and a consumed
//         slot is "reposted" by advancing only the RQ doorbell record.
//   WQE:  regular post-recv (the device-side equivalent of mrc_post_recv): reserve the RQ slot,
//         rewrite its WQE, mark it ready and submit it through the RQ doorbell path.
#define NCCL_GIN_MRC_RQ_POST_MODE_DBR 0
#define NCCL_GIN_MRC_RQ_POST_MODE_WQE 1

// Each context owns one signal table, [nSignals] slots indexed by NCCL signal id. The NIC writes no
// slot (a bare signal is a zero-length Write-with-Immediate), so the table has no MR: the device
// drain applies every signal.

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

  // Receive drain of the QP carrying this peer's traffic to us: consumer_index is the
  // running count of consumed receive CQEs.
  uint64_t consumer_index;
  uint32_t rq_depth;
  int32_t status;     // latched receive CQ error (0 = healthy)
  uint32_t post_mode; // NCCL_GIN_MRC_RQ_POST_MODE_*
  // RDW get responses consumed from this peer, in request order, and the first non-OK response
  // status / protocol error (0 = healthy).
  uint64_t get_completed;
  int32_t get_status;
  // Partially received multi-chunk ncclGin_SignalAdd from this peer. Only the drain-lock holder
  // touches these.
  uint32_t add_shift; // bit position of the next chunk
  uint64_t add_acc;   // sum of the chunks received so far
};

// Registered windows of this rank, used by the RDW server to validate a get request's source
// key and bounds and to translate the requester-supplied rkey into the local lkey.
#define NCCL_GIN_MRC_MR_TABLE_CAPACITY 1024
// RDW get wire constants shared with the host plugin (which sizes the RDW buffer).
#define NCCL_GIN_MRC_RDW_DESC_BYTES 128 // request descriptor and request-ring slot size
#define NCCL_GIN_MRC_RDW_GEN_MASK 0x3Fu // generation bits carried in the immediate
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
  // The QP to each peer: an array of GPU pointers to the DOCA SDK doca_gpu_dev_verbs_qp each QP was
  // exported to (doca_gpu_verbs_export_qp).
  void** sdk_qps;
  struct ncclGinMrcPeerState* peers; // [nRanks]

  // This context's signal table, indexed by signal id. The device drain owns the slots.
  uint64_t* signals;
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
  uint32_t rdw_gen;        // generation tag carried by requests and echoed in responses

  // An MRC QP cannot target itself, so the QP to the local rank is connected to a
  // dedicated loopback responder QP: puts and signals to self are posted on the local rank's QP and
  // their receive completions arrive on the responder. self_rq_qp is the responder (DOCA SDK
  // doca_gpu_dev_verbs_qp), whose receive queue the drain uses for the local rank. Its send queue
  // carries no traffic, so it also hosts the Data Direct flushes below. NULL when self_loop_off.
  void* self_rq_qp;
  // Data Direct NICs write registered windows over a direct PCIe link while CQEs reach
  // GPU memory over C2C, so a receive CQE does not imply its data is visible. When dd_flush is set,
  // the drain posts a DUMP to the 8-byte buffer dd_flush_lkey (registered for Data Direct) on
  // self_rq_qp and waits for it before publishing signals or get completions.
  uint32_t dd_flush;
  uint32_t dd_flush_lkey; // big-endian
};

struct ncclGinMrcMemHandle {
  uint32_t* rkeys; // [nRanks], big-endian
  uint32_t lkey;   // big-endian
};

#endif /* _NCCL_DEVICE_GIN_MRC_DEV_H_ */
