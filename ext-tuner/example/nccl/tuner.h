/*************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2023, Meta Platforms, Inc. and affiliates.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_TUNER_H_
#define NCCL_TUNER_H_

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include "nccl.h"

typedef enum {NCCL_LOG_NONE=0, NCCL_LOG_VERSION=1, NCCL_LOG_WARN=2, NCCL_LOG_INFO=3, NCCL_LOG_ABORT=4, NCCL_LOG_TRACE=5} ncclDebugLogLevel;
typedef enum {NCCL_INIT=1, NCCL_COLL=2, NCCL_P2P=4, NCCL_SHM=8, NCCL_NET=16, NCCL_GRAPH=32, NCCL_TUNING=64, NCCL_ENV=128, NCCL_ALLOC=256, NCCL_CALL=512, NCCL_PROXY=1024, NCCL_NVLS=2048, NCCL_ALL=~0} ncclDebugLogSubSys;

typedef void (*ncclDebugLogger_t)(ncclDebugLogLevel level, unsigned long flags, const char *file, int line, const char *fmt, ...);

#define NCCL_NUM_FUNCTIONS 5 // Send/Recv not included for now
typedef enum { ncclFuncBroadcast, ncclFuncReduce, ncclFuncAllGather, ncclFuncReduceScatter, ncclFuncAllReduce, ncclFuncSendRecv, ncclFuncSend, ncclFuncRecv, ncclNumFuncs} ncclFunc_t;

#define NCCL_NUM_ALGORITHMS 6 // Tree/Ring/CollNet*
#define NCCL_ALGO_UNDEF -1
#define NCCL_ALGO_TREE 0
#define NCCL_ALGO_RING 1
#define NCCL_ALGO_COLLNET_DIRECT 2
#define NCCL_ALGO_COLLNET_CHAIN 3
#define NCCL_ALGO_NVLS 4
#define NCCL_ALGO_NVLS_TREE 5

#define NCCL_NUM_PROTOCOLS 3 // Simple/LL/LL128
#define NCCL_PROTO_UNDEF -1
#define NCCL_PROTO_LL 0
#define NCCL_PROTO_LL128 1
#define NCCL_PROTO_SIMPLE 2

// API to be implemented by external tuner
typedef struct {
  // Name of the tuner
  const char* name;

  // Initializes tuner states.
  // Inputs:
  //   - nRanks: number of ranks in current communicator. Each communicator initialize its own tuner.
  //   - nNodes: number of nodes in current communicator.
  //   - logFunction: a logFunction can be useful to integrate logging together with NCCL core.
  // Outputs:
  //   - context: tuner context object
  ncclResult_t (*init)(size_t nRanks, size_t nNodes, ncclDebugLogger_t logFunction, void **context);

  // Gets info (algo, protocol, number of ctas and threads) for a given collective.
  // Inputs:
  //   - context: tuner context object
  //   - collType: collective type , e.g., allreduce, allgather…
  //   - nBytes: collective size in bytes
  //   - collNetSupport: whether collnet supports this type
  //   - nvlsSupport: whether nvlink sharp supports this time
  //   - numPipeOps: number of operations in the group
  //
  // Outputs:
  //   - algorithm: selected algorithm to be used for the given collective
  //   - protocol: selected protocol to be used for the given collective
  //   - nChannels: number of channels (hence SMs) to be used.
  //
  // If getCollInfo() does not return ncclSuccess, NCCL will fall back to the
  // default tuning for the given collective.
  // Also, the plugin is allowed to not set any output, or set only the
  // algorithm and protocol, but not only the algorithm or only the protocol.
  // Unset fields will be set automatically by NCCL.
  ncclResult_t (*getCollInfo)(void* context, ncclFunc_t collType, size_t nBytes,
                              int collNetSupport, int nvlsSupport, int numPipeOps,
                              int *algorithm, int *protocol, int* nChannels);

  // Terminates the plugin and cleans up any resources that the plugin allocated.
  // context: tuner context object
  ncclResult_t (*destroy)(void* context);
} ncclTuner_v2_t;

typedef ncclTuner_v2_t ncclTuner_t;

#define NCCL_TUNER_PLUGIN_SYMBOL "ncclTunerPlugin_v2"

// Latencies in us, Bandwidths in GB/s
// Tree { LL, LL128, Simple } , Ring { LL, LL128, Simple }
// Base algorithm latencies
static const float nccl_base_lat[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS] = {
        {  6.8, 14.0,    0 }, // Tree
        {  6.6, 14.0,  8.4 }, // Ring
        {    0,    0,    0 }, // Collnet Direct
        {    0,    0,    0 }, // Collne Chain
        {    0,    0,    0 }, // NVLS
        {    0,    0,    0 }  // NVLS Tree
};

// NVLink
static float nccl_nvlink_lat[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS] = {
       { .6, 1.25,  28 }, /* Tree (LL/LL128/Simple)*/
       { .6,  1.9, 3.4 }, /* Ring (LL/LL128/Simple)*/
       {  0,    0, 3.7 }, /* CollNetDirect (Simple)*/
       {  0,    0, 2.8 }, /* CollNetChain (Simple)*/
       {  0,    0,  25 }, /* NVLS (Simple) */
       {  0,    0,  25 }  /* NVLSTree (Simple) */
};

#define NCCL_TUNER_NET_LAT			(3)
#define NCCL_TUNER_NET_NUM_CHANNELS	(16)
#define NCCL_TUNER_INTERNODE_BW		(50.0 * 1024ULL * 1024ULL * 1024ULL * 1e-6)

/*
 * For Hopper GPUs, all intranode communication goes over NVLink, so use
 * the bandwidth for SM90 architecture in NCCL (SM90_NVLINK_BW).
 *
 * This is unidirectional bandwidth per NVLink (900GB/s bidirectional on the
 * platform, with 18 NVLinks in total. NCCL considers a 20% protocol overhead,
 * leaving 20GB/s bandwidth per link).
 */
#define NCCL_TUNER_INTRANODE_BW		(20.0 * 1024 * 1024 * 1024 * 1e-6)

struct nccl_tuner_model_params {
    float net_lat;
	float internode_bw;
	float intranode_bw;
	int num_channels;
};

struct nccl_tuner_model_dims {
	int num_ranks;
	int num_nodes;
};

struct nccl_tuner_context {
	struct nccl_tuner_model_dims dims;
	struct nccl_tuner_model_params params;
};


static long log2i(long n) {
	long l = 0;
	while (n>>=1) l++;
	return l;
}

float nccl_tuner_compute_cost(struct nccl_tuner_model_params *params, struct nccl_tuner_model_dims *dims,
                                  ncclFunc_t func, int algo, int proto, int pipe_ops, size_t size)
{
	float cost = -1;
	float latency = 0;
	float bw = 0;
	float intraLat = 0;
	float interLat = 0;
	int num_steps = 0;
	int num_internode_steps = 0;
	int num_intranode_steps = 0;

	latency = nccl_base_lat[algo][proto];
	intraLat = nccl_nvlink_lat[algo][proto];
	interLat = params->net_lat;

	// Also add the flush extra latency
    //if (p == NCCL_PROTO_SIMPLE) interLat += graphs[a]->latencyInter;

	switch(func) {
	case ncclFuncAllReduce:
		switch(algo) {
		case NCCL_ALGO_RING:
			num_steps = 2 * (dims->num_ranks - 1);
			num_internode_steps = 2 * dims->num_nodes;
			num_intranode_steps = num_steps - num_internode_steps;
			latency += num_internode_steps * interLat + num_intranode_steps * intraLat;
			bw = params->internode_bw * params->num_channels;
			break;

		case NCCL_ALGO_TREE:
			latency += 2 * (((dims->num_ranks / dims->num_nodes) - 1) * intraLat
					+ log2i(dims->num_nodes) * interLat);
			bw = params->internode_bw * params->num_channels / 2;
			break;

		case NCCL_ALGO_NVLS_TREE:
			latency += intraLat + 2 * log2i(dims->num_nodes) * interLat;
			bw = params->internode_bw * params->num_channels / 2;
			break;

		default:
			printf("Algorithm %d for collective %d without a model.", algo, func);
			return -1;
		}
		break;

	default:
		return -1;
	}

	/* Penalize the low-latency protocol bandwidths for their overhead */
	if (proto == NCCL_PROTO_LL)
		/* 8B total with 4B data and 4B flags, so take a 50% hit */
		bw *= 0.5;
	else if (proto == NCCL_PROTO_LL128)
		/* 120B data and 8B flags */
		bw *= 0.9375;

	// cost model calculation
	cost = (latency * pipe_ops) + size / bw;

	return cost;
}

#endif
