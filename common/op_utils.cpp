#include <cstddef>
#include <cstdint>
#include <type_traits> // for std::enable_if_t
#include "counters.h"
#include "my_debug.h"
#include "op_utils.h"
#include "data.h"
#include "intermittent-cnn.h"
#include "my_dsplib.h"
#include "cnn_common.h"
#include "platform.h"
#include "counters.h"

// Not using DSPLIB_DATA here as it does not work under C++ (?)
#ifdef __MSP430__
#pragma DATA_SECTION(".leaRAM")
#endif
NOINIT int16_t lea_buffer[LEA_BUFFER_SIZE];

NOINIT int16_t op_buffer[OP_BUFFER_LEN];
static_assert(OUTPUT_LEN <= OP_BUFFER_LEN, "invalid OP buffer size");

#if HAWAII
static uint32_t non_recorded_jobs = 0;
void hawaii_record_footprints(Model* model, uint32_t vector_len) {
    non_recorded_jobs += vector_len;
#if 0
    for (; non_recorded_jobs >= BATCH_SIZE; non_recorded_jobs -= BATCH_SIZE) {
        write_hawaii_layer_footprint(model->layer_idx, BATCH_SIZE);
    }
#else
    write_hawaii_layer_footprint(model->layer_idx, non_recorded_jobs / BATCH_SIZE * BATCH_SIZE);
    non_recorded_jobs %= BATCH_SIZE;
#endif
}
#endif

int16_t upper_gauss(int16_t a, int16_t b) {
    return (a + b - 1) / b;
}

void float_to_scale_params(int16_t *scaleFract, uint8_t *shift, const Scale& scale) {
    float_to_scale_params(scaleFract, shift, scale.toFloat());
}

void float_to_scale_params(int16_t *scaleFract, uint8_t *shift, float scale) {
    MY_ASSERT(scale > 0);
    *shift = 0;
    while (scale >= 1) {
        scale /= 2;
        (*shift)++;
    }
    *scaleFract = scale * 32768;
}

uint8_t count_dims(const ParameterInfo* data) {
    uint8_t dim_idx = 0;
    while (dim_idx < MAX_NUM_DIMS && data->dims[dim_idx]) {
        dim_idx++;
    }
    return dim_idx;
}

void recalculate_params_len(ParameterInfo* output) {
    output->params_len = sizeof(int16_t);
    for (uint8_t dim_idx = 0; (dim_idx < MAX_NUM_DIMS) && output->dims[dim_idx]; dim_idx++) {
        output->params_len *= output->dims[dim_idx];
    }
}

void iterate_chunks(Model *model, const ParameterInfo *param, uint16_t start_offset, uint16_t len, const ChunkHandler& chunk_handler, void* params) {
    uint16_t params_len;
    if (!len) {
        params_len = param->params_len / sizeof(int16_t);
    } else {
        params_len = start_offset + len;
    }
    uint16_t chunk_len = LIMIT_DMA_SIZE((LEA_BUFFER_SIZE - 1) / 2 * 2);
    uint8_t state_bit = 0;

    uint16_t cur_chunk_len;
    for (uint32_t offset = start_offset; offset < params_len; offset += cur_chunk_len) {
        cur_chunk_len = MIN_VAL(chunk_len, params_len - offset);
        MY_ASSERT(cur_chunk_len != 0);
        chunk_handler(offset, cur_chunk_len, state_bit, params);
    }
}


void fix_first_unfinished_value_offset(const Model* model, uint32_t* p_first_unfinished_value_offset) {
    if (BATCH_SIZE >= 2) {
        return;
    }
    // Force recovery from an even OFM index as most DSPLib function does not like odd dimensions
    if (*p_first_unfinished_value_offset % 2) {
        (*p_first_unfinished_value_offset)--;
#if HAWAII
        write_hawaii_layer_footprint(model->layer_idx, -1); // discard last job
#endif
    }
}

void make_buffer_aligned(int16_t** p_buffer) {
    if ((*p_buffer - lea_buffer) % 2) {
        (*p_buffer)++;
    }
}

float q15_to_float(int16_t val, const ValueInfo& val_info, uint8_t* p_use_prefix, bool has_state) {
    return val_info.scale * static_cast<int32_t>(val) / 32768.0;
}

void my_offset_q15_batched(const int16_t *pSrc, int16_t offset, int16_t *pDst, uint32_t blockSize, bool enforce_states) {
    MY_ASSERT(pSrc == pDst);
    if (BATCH_SIZE == 1) {
        my_offset_q15(pSrc, offset, pDst, blockSize);
    } else {
        for (uint32_t val_idx = BATCH_SIZE - 1; val_idx < blockSize; val_idx += BATCH_SIZE) {
            pDst[val_idx] += offset;
        }
    }
}
