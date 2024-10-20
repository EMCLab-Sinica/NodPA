#pragma once

#include <cstdint>
#include "data.h"
#include "platform.h"

#define OP_BUFFER_LEN 512

struct Model;
struct ParameterInfo;
struct SlotInfo;
struct ValueInfo;
struct Scale;

typedef void (*ChunkHandler)(uint32_t output_offset, uint16_t output_chunk_len, int8_t old_output_state_bit, void* params);

extern int16_t lea_buffer[LEA_BUFFER_SIZE];
int16_t upper_gauss(int16_t a, int16_t b);
void float_to_scale_params(int16_t *scaleFract, uint8_t *shift, float scale);
void float_to_scale_params(int16_t *scaleFract, uint8_t *shift, const Scale& scale);
uint8_t count_dims(const ParameterInfo* data);
void recalculate_params_len(ParameterInfo* output);
void iterate_chunks(Model *model, const ParameterInfo *param, uint16_t start_offset, uint16_t len, const ChunkHandler& callback, void* params);
void determine_tile_c(ParameterInfo *param, const ParameterInfo* input, const ParameterInfo *filter = nullptr);

#if HAWAII
void hawaii_record_footprints(Model* model, uint32_t vector_len);
#endif

void fix_first_unfinished_value_offset(const Model* model, uint32_t* p_first_unfinished_value_offset);
void make_buffer_aligned(int16_t** p_buffer);
float q15_to_float(int16_t val, const ValueInfo& val_info, uint8_t* p_use_prefix = nullptr, bool has_state = true);
void my_offset_q15_batched(const int16_t *pSrc, int16_t offset, int16_t *pDst, uint32_t blockSize, bool enforce_states = false);

extern int16_t op_buffer[OP_BUFFER_LEN];
