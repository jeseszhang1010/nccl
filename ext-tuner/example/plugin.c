/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "tuner.h"


ncclResult_t pluginInit(size_t nRanks, size_t nNodes, ncclDebugLogger_t logFunction, void **context) {

	struct nccl_tuner_context *nccl_tuner_ctx;

	const struct nccl_tuner_model_params params = {
		.net_lat = NCCL_TUNER_NET_LAT,
		.internode_bw = NCCL_TUNER_INTERNODE_BW,
		.intranode_bw = NCCL_TUNER_INTRANODE_BW,
		.num_channels = NCCL_TUNER_NET_NUM_CHANNELS
	};

	nccl_tuner_ctx = calloc(1, sizeof(struct nccl_tuner_context));
	if(!nccl_tuner_ctx) {
		return ncclInternalError;
	}

	nccl_tuner_ctx->dims.num_ranks = nRanks;
	nccl_tuner_ctx->dims.num_nodes = nNodes;
	nccl_tuner_ctx->params = params;

	*context = (void*)nccl_tuner_ctx;

	return ncclSuccess;
}

ncclResult_t pluginGetCollInfo(void* context, ncclFunc_t collType, size_t nBytes,
                              int collNetSupport, int nvlsSupport, int numPipeOps,
                              int *algorithm, int *protocol, int* nChannels) {
	float cost = 0;
	float lowest = FLT_MAX;
	int algo, proto = 0;
	struct nccl_tuner_context *nccl_tuner_ctx = (struct nccl_tuner_context *)context;

	if(nccl_tuner_ctx->dims.num_nodes <= 2)
		return ncclSuccess;

	for (algo = 0; algo < NCCL_NUM_ALGORITHMS; algo++) {
		/* Skip NVLS used only for single-node jobs */
		if (algo == NCCL_ALGO_NVLS)
			continue;

		if (!nvlsSupport && (algo == NCCL_ALGO_NVLS_TREE))
			continue;

		if (!collNetSupport && (algo == NCCL_ALGO_COLLNET_DIRECT || algo == NCCL_ALGO_COLLNET_CHAIN))
			continue;

		for (proto = 0; proto < NCCL_NUM_PROTOCOLS; proto++) {
			/* Not a supported combination in NCCL */
			if (algo == NCCL_ALGO_NVLS_TREE && proto != NCCL_PROTO_SIMPLE)
				continue;

			cost = nccl_tuner_compute_cost(&nccl_tuner_ctx->params, &nccl_tuner_ctx->dims,
							   collType, algo, proto, numPipeOps, nBytes);
			if (cost < 0)
				continue;

			if (cost < lowest) {
				*algorithm = algo;
				*protocol = proto;
				lowest = cost;
			}
		}
	}
	return ncclSuccess;
}

ncclResult_t pluginDestroy(void* context) {
	if(context != NULL) {
		free(context);
	}
	return ncclSuccess;
}

#define PLUGIN_NAME "azure-tuner"

const ncclTuner_v2_t ncclTunerPlugin_v2 = {
  .name = PLUGIN_NAME,
  .init = pluginInit,
  .getCollInfo = pluginGetCollInfo,
  .destroy = pluginDestroy
};
