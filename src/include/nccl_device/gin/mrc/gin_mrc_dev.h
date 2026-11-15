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
// has a receive queue. Who drains it is chosen when NCCL is built (GIN_MRC_RQ), as the two ways use
// different DOCA GPUNetIO device code:
//   CPU RQ (NCCL_GIN_MRC_GPU_RQ=0): device code from the open-source GPUNetIO (DOCA_HOME) posts
//     with doca_gpu_dev_verbs_put_immediate; the plugin drains the receive queues on the CPU
//     (doca_gpu_verbs_rq_progress) into per-context DOCA GPUNetIO signal maps.
//   GPU RQ (NCCL_GIN_MRC_GPU_RQ=1): the plugin exports its QPs to the DOCA SDK
//     (doca_gpu_verbs_export_sdk_qp), and device code built against the DOCA SDK device headers
//     posts the receive WQEs and drains the receive CQs itself.
// NCCL tells the plugin which one its device code implements through the backend version it passes
// to createContext, which is also the layout of the structs below that the device code reads.
#define NCCL_GIN_MRC_VERSION_CPU_RQ 7
#define NCCL_GIN_MRC_VERSION_GPU_RQ 8

// How receive WQEs are posted/reposted in GPU RQ mode (ncclGinMrcPeerState::post_mode). Selected by
// the host plugin at runtime via NCCL_GIN_MRC_RQ_POST_MODE.
//   DBR:  lightweight. The ring is filled once with immutable notification WQEs and a consumed
//         slot is "reposted" by advancing only the RQ doorbell record.
//   WQE:  regular post-recv (the device-side equivalent of mrc_post_recv): reserve the RQ slot,
//         rewrite its WQE, mark it ready and submit it through the RQ doorbell path.
#define NCCL_GIN_MRC_RQ_POST_MODE_DBR 0
#define NCCL_GIN_MRC_RQ_POST_MODE_WQE 1

// Signal ids of a context in CPU RQ mode. The signal map adds each immediate's value to the GPU slot
// registered for its id. Each context owns one signal table, indexed by id:
//   [0, nSignals)                 NCCL indexed signals
//   nSignals                      sink: payload target of bare-signal writes (not in the map)
//   rdw_signal_base + p           RDW get requests received from rank p
//   rdw_signal_base + nRanks + p  RDW get responses from rank p (OK)
//   rdw_signal_base + 2*nRanks+p  RDW get responses from rank p (error)
// with rdw_signal_base = nSignals + 1. Every rank uses the same layout, so a sender addresses the
// receiver's slot by id alone. In GPU RQ mode the table is only [0, nSignals] (signals and sink).
#define NCCL_GIN_MRC_SIGNAL_TABLE_SIZE(nSignals, nRanks) ((nSignals) + 1 + 3 * (nRanks))

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

  // CPU RQ mode. RDW get, server side: requests from this peer served so far. Requests are served
  // in order, and the peer's request counter (signal rdw_signal_base + peer) says how many arrived.
  uint64_t rdw_served;

  // GPU RQ mode. Receive drain of the QP carrying this peer's traffic to us: consumer_index is the
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
#define NCCL_GIN_MRC_RDW_GEN_MASK 0x3Fu // GPU RQ mode: generation bits carried in the immediate
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
  // The QP to each peer. CPU RQ mode: qps is an array of the open-source GPUNetIO
  // doca_gpu_dev_verbs_qp, one per peer. GPU RQ mode: sdk_qps is an array of GPU pointers to the
  // DOCA SDK doca_gpu_dev_verbs_qp each QP was exported to (doca_gpu_verbs_export_sdk_qp).
  void* qps;
  void** sdk_qps;
  struct ncclGinMrcPeerState* peers; // [nRanks]

  // This context's signal table (id layout above), indexed by signal id.
  uint64_t* signals;
  // CPU RQ mode: [nSignals] reset baselines. The host signal map is the only writer of signal
  // slots and adds to a host-side shadow of each, so a device-side store would be overwritten; a
  // reset records the slot here instead, and the logical value of signal i is
  // signals[i] - signal_base[i]. GPU RQ mode: NULL, the device drain owns the slots.
  uint64_t* signal_base;
  uint32_t* signal_rkeys; // [nRanks] big-endian rkeys of the peers' signal MRs
  // Element index, within the signal MR, of this context's sink slot. MRC rejects zero-length
  // Write-with-Immediate, so bare signals write 8 bytes there.
  uint32_t signals_sink_offset;
  uint32_t rdw_signal_base; // CPU RQ mode, see the id layout above

  uint64_t* counters; // local indexed counters, maintained by the send-CQ drain

  uint32_t sink_lkey; // big-endian lkey of a local scratch buffer we write from but never read

  // Receiver-Driven Write (RDW) get. MRC has no RDMA Read, so a get is a request descriptor
  // sent by Write-with-Immediate into the data owner's request ring; the owner's device progress
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
  uint32_t rdw_gen;        // GPU RQ mode: generation tag carried by requests and echoed in responses

  // GPU RQ mode. An MRC QP cannot target itself, so the QP to the local rank is connected to a
  // dedicated loopback responder QP: puts and signals to self are posted on the local rank's QP and
  // their receive completions arrive on the responder. self_rq_qp is the responder (DOCA SDK
  // doca_gpu_dev_verbs_qp), whose receive queue the drain uses for the local rank. Its send queue
  // carries no traffic, so it also hosts the Data Direct flushes below.
  void* self_rq_qp;
  // GPU RQ mode. Data Direct NICs write registered windows over a direct PCIe link while CQEs reach
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
