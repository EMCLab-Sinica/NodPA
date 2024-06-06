#pragma once

#include <cstddef>
#include <cstdint>
#include "dynbal-config.h"
#include "layer-defs.h"

#define OFM_DUMPED 1

struct Node;
struct NodeFlags;
struct ParameterInfo;

const Node* get_node(size_t i);
const NodeFlags* get_node_orig_flags(size_t i);
const Node* get_node(const ParameterInfo* param);

CurNodeFlags* get_node_flags(uint16_t node_idx);
void commit_node_flags(const NodeFlags* node_flags);
