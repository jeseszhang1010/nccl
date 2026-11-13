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

// How receive WQEs are posted/reposted (ncclGinMrcRqState::post_mode). Selected by the host
// plugin at runtime via NCCL_GIN_MRC_RQ_POST_MODE.
//   DBR:  lightweight. The ring is filled once with immutable notification WQEs and a consumed
//         slot is "reposted" by advancing only the RQ doorbell record.
//   WQE:  regular post-recv (the device-side equivalent of mrc_post_recv): reserve the RQ slot,
//         rewrite its WQE, mark it ready and submit it through the RQ doorbell path.
#define NCCL_GIN_MRC_RQ_POST_MODE_DBR 0
#define NCCL_GIN_MRC_RQ_POST_MODE_WQE 1

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
  uint32_t post_mode; // NCCL_GIN_MRC_RQ_POST_MODE_*
  // Requester-side RDW get state for this peer (see ncclGinMrcGPUContext::rdw_*). Request ids
  // are allocated under get_lock in wire order, and the peer serves and answers them in that
  // order, so get_completed is a monotonic count of answered requests.
  uint64_t get_issued;    // requests posted to this peer
  uint64_t get_completed; // responses consumed (advanced by the receive drain)
  int32_t get_status;     // latched first non-OK response status / protocol error (0 = healthy)
  int get_lock;           // serializes request posting to this peer (0=free, 1=held)
  // Partially received multi-chunk ncclGin_SignalAdd from this peer (see encodeSignal in
  // gin_mrc.h). Only the drain-lock holder touches these.
  uint64_t add_acc;   // sum of the chunks received so far
  uint32_t add_shift; // bit position of the next chunk
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

  // Receiver-Driven Write (RDW) get. MRC has no RDMA Read, so a get is a request descriptor
  // sent by Write-with-Immediate into the data owner's request ring; the owner's receive drain
  // serves it with RDMA Writes plus a Write-with-Immediate response on the same QP.
  //
  // RDW MR layout, one region per context (identical on every rank):
  //   [0, R)        request ring  [nRanks requester][rdw_window] x 128 B (written by peers)
  //   [R, 2R)       staging       [nRanks peer][rdw_window] x 128 B (local descriptors)
  //   [2R, 2R+128)  response landing for the 8 B response payload
  // where R = nRanks * rdw_window * 128.
  struct ncclGinMrcMrTable* mr_table; // this rank's registered windows
  uint8_t* rdw_region;     // device pointer to this context's region of the local RDW MR
  uint32_t* rdw_rkeys;     // [nRanks] big-endian rkeys of the peers' RDW MRs
  uint64_t rdw_ctx_offset; // offset of this context's region within the (zero-based) RDW MR
  uint32_t rdw_lkey;       // big-endian
  uint32_t rdw_window;     // in-flight requests per peer (power of two)
  uint32_t rdw_gen;        // generation tag carried by requests and echoed in responses

  // An MRC QP cannot target itself, so qps[rank] is connected to a dedicated loopback responder
  // QP: puts and signals to self are posted on qps[rank] and their receive completions arrive on
  // the responder. self_rq_qp is that responder (doca_gpu_dev_verbs_qp), whose receive queue the
  // drain uses for the local rank. Its send queue carries no traffic, so it also hosts the Data
  // Direct flushes below.
  void* self_rq_qp;
  // Data Direct NICs write registered windows over a direct PCIe link while CQEs reach GPU memory
  // over C2C, so a receive CQE does not imply its data is visible. When dd_flush is set, the drain
  // posts a DUMP to the 8-byte buffer dd_flush_lkey (registered for Data Direct) on self_rq_qp and
  // waits for it before publishing signals or get completions.
  uint32_t dd_flush;
  uint32_t dd_flush_lkey; // big-endian
};

struct ncclGinMrcMemHandle {
  uint32_t* rkeys; // [nRanks], big-endian
  uint32_t lkey;   // big-endian
};

#endif /* _NCCL_DEVICE_GIN_MRC_DEV_H_ */
