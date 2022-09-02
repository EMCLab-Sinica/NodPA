from __future__ import annotations

import dataclasses
import logging
from typing import Any

import onnx
import tensorly
import numpy as np
import numpy.typing

from utils import (
    run_model,
    ModelData,
)
from fine_tune import (
    fine_tune,
)

ACCURACY_LOSS_THRESHOLD = 0.03

@dataclasses.dataclass(frozen=True)
class Context:
    model: onnx.ModelProto
    node: onnx.NodeProto
    node_idx: int
    orig_weights: onnx.TensorProto
    orig_weights_data: numpy.typing.ArrayLike
    train_data: ModelData
    test_data: ModelData
    orig_accuracy: float
    epoch: int
    config: dict[str, Any]

logger = logging.getLogger('decomposition')

def add_new_nodes(ctx, model, new_nodes):
    model.graph.node.remove(ctx.node)
    model.graph.initializer.remove(ctx.orig_weights)
    node_idx = ctx.node_idx
    for new_node in new_nodes:
        model.graph.node.insert(node_idx, new_node)
        node_idx += 1

def try_cp_rank(ctx, try_rank, add_cp_filters_func):
    logger.info('Try CP rank=%d', try_rank)

    new_nodes = []
    orig_weights_shape = np.shape(ctx.orig_weights_data)
    weights_data = np.squeeze(ctx.orig_weights_data)
    weights, factors = tensorly.decomposition.parafac(weights_data, rank=try_rank)

    decomposed_model = onnx.ModelProto()
    decomposed_model.ParseFromString(ctx.model.SerializeToString())

    add_cp_filters_func(ctx, decomposed_model, weights, factors, orig_weights_shape, new_nodes)
    add_new_nodes(ctx, decomposed_model, new_nodes)

    decomposed_model = fine_tune(decomposed_model, ctx.train_data, ctx.test_data, ctx.epoch, qat=False, config=ctx.config)
    accuracy = run_model(decomposed_model, ctx.test_data, limit=None, verbose=False)
    logger.info('Accuracy: %.4f', accuracy)

    return accuracy, decomposed_model

def find_rank(orig_accuracy, max_dim, try_rank_func):
    rank_lower = 2
    rank_upper = 2

    candidates = []
    candidates_lower_accuracy = []

    while rank_upper <= max_dim:
        accuracy, decomposed_model = try_rank_func(try_rank=rank_upper)
        if accuracy >= orig_accuracy - ACCURACY_LOSS_THRESHOLD:
            logger.debug('%f >= %f - %f', accuracy, orig_accuracy, ACCURACY_LOSS_THRESHOLD)
            candidates.append((accuracy, rank_upper, decomposed_model))
            break
        logger.debug('%f < %f - %f', accuracy, orig_accuracy, ACCURACY_LOSS_THRESHOLD)
        candidates_lower_accuracy.append((accuracy, rank_upper, decomposed_model))
        rank_lower = rank_upper
        rank_upper *= 2

    rank_upper = min(rank_upper, max_dim)

    # Only try even ranks to avoid matrices with odd columns in GEMM
    while rank_lower + 2 < rank_upper:
        try_rank = ((rank_upper + rank_lower) // 2 + 1) // 2 * 2
        accuracy, decomposed_model = try_rank_func(try_rank=try_rank)
        if accuracy >= orig_accuracy - ACCURACY_LOSS_THRESHOLD:
            logger.debug('%f >= %f - %f', accuracy, orig_accuracy, ACCURACY_LOSS_THRESHOLD)
            rank_upper = try_rank
            candidates.append((accuracy, try_rank, decomposed_model))
        else:
            logger.debug('%f < %f - %f', accuracy, orig_accuracy, ACCURACY_LOSS_THRESHOLD)
            candidates_lower_accuracy.append((accuracy, try_rank, decomposed_model))
            rank_lower = try_rank

    if candidates:
        candidates.sort(key=lambda item: item[1])
        accuracy, rank, decomposed_model = candidates[0]
    else:
        candidates_lower_accuracy.sort(key=lambda item: item[0], reverse=True)
        accuracy, rank, decomposed_model = candidates_lower_accuracy[0]
    logger.debug('Using rank %d with accuracy %f', rank, accuracy)

    return decomposed_model
