#pragma once

#include <cstdint>
#include "cnn_common.h"
#include "data.h"
#include "my_debug.h"

struct ParameterInfo;
struct Model;

uint32_t job_index_to_offset(const ParameterInfo* output, uint32_t job_index);
uint32_t batch_start(uint32_t batch_end_offset);
uint32_t run_recovery(Model *model, ParameterInfo *output);
