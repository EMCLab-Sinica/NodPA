#pragma once

#include <cstdint>
#include "cnn_common.h"
#include "data.h"
#include "my_debug.h"

struct ParameterInfo;
struct Model;

uint32_t job_index_to_offset(const ParameterInfo* output, uint32_t job_index);
uint32_t batch_start(uint32_t batch_end_offset);

int8_t get_state_bit(Model *model, uint8_t slot_id);

#if HAWAII
static inline bool offset_has_state(uint32_t offset) {
    return offset % BATCH_SIZE == BATCH_SIZE - 1;
}
#else
static inline bool offset_has_state(uint32_t) {
    return false;
}
#endif

int8_t param_state_bit(Model *model, const ParameterInfo *param, uint16_t offset);

uint32_t run_recovery(Model *model, ParameterInfo *output);
