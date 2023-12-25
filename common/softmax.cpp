#include <cmath>

#include "DSPLib.h"

#include "cnn_common.h"
#include "my_debug.h"
#include "my_dsplib.h"
#include "op_utils.h"
#include "platform.h"

void alloc_softmax(Model* model, const ParameterInfo* input[], ParameterInfo* output, const Node* node, CurNodeFlags* node_flags, const NodeFlags*) {
    int axis = node_flags->softmax.axis;
    if (axis == 1) {
        SlotInfo *cur_slot_info = get_slot_info(model, output->slot);
        if (cur_slot_info) {
            cur_slot_info->user = model->layer_idx;
        }
    } else if (axis == 2) {
        output->slot = get_next_slot(model, input[0]);
    } else {
        MY_ASSERT(false);
    }
}

void handle_softmax(Model* model, const ParameterInfo* input[], ParameterInfo* output, const Node* node, CurNodeFlags* node_flags, const NodeFlags*) {
    int axis = node_flags->softmax.axis;

    if (axis == 1) {
        // Do nothing - softmax does not change the relative order of values.
        // Just let run_model determine the max value
    } else if (axis == 2) {
        const ParameterInfo* X = input[0];
        const uint16_t softmax_length = X->dims[axis];

        uint16_t idx0 = 0, idx1 = 0, idx3 = 0;
        const uint16_t idx3_tile_size = 2; // TODO: make the tile size configurable
#if HAWAII
        read_softmax_loop_indices(&idx0, &idx1, &idx3);
#endif

        for (; idx0 < X->dims[0];) {
            for (; idx1 < X->dims[1];) {
                for (; idx3 < X->dims[3];) {
                    uint32_t softmax_vector_base_offset = idx0 * X->dims[1] * X->dims[2] * X->dims[3] + idx1 * X->dims[2] * X->dims[3] + idx3;

                    for (uint16_t softmax_idx = 0; softmax_idx < softmax_length; softmax_idx++) {
                        uint32_t softmax_value_offset = softmax_vector_base_offset + softmax_idx * X->dims[3];
                        my_memcpy_from_param(model, op_buffer, X, softmax_value_offset, idx3_tile_size * sizeof(int16_t));
                        for (uint16_t intra_tile_idx3 = 0; intra_tile_idx3 < idx3_tile_size; intra_tile_idx3++) {
                            lea_buffer[intra_tile_idx3 * softmax_length + softmax_idx] = op_buffer[intra_tile_idx3];
                        }
                    }

                    for (uint16_t intra_tile_idx3 = 0; intra_tile_idx3 < idx3_tile_size; intra_tile_idx3++) {
                        int16_t* softmax_buffer = lea_buffer + intra_tile_idx3 * softmax_length;

                        float softmax_sum = 0.0f; // the denominator in softmax equation
                        // avoid exponential overflow and underflow
                        // https://www.cnblogs.com/guoyaohua/p/8900683.html
                        int16_t max_val = 0;
                        uint16_t max_val_idx = 0;
                        my_max_q15(softmax_buffer, softmax_length, &max_val, &max_val_idx);
                        my_offset_q15(softmax_buffer, -(max_val), softmax_buffer, softmax_length);

                        for (uint16_t softmax_idx = 0; softmax_idx < softmax_length; softmax_idx++) {
                            // exponentials
                            float val = q15_to_float(softmax_buffer[softmax_idx], ValueInfo(output));
                            val = std::exp(val);
                            softmax_buffer[softmax_idx] = _Q15(val / output->scale.toFloat());
                            softmax_sum += val;
                        }

                        // normalize the row
                        int16_t scaleFract;
                        uint8_t shift;
                        float_to_scale_params(&scaleFract, &shift, 1.0f/softmax_sum);
                        my_scale_q15(softmax_buffer, scaleFract, shift, softmax_buffer, softmax_length);
                    }

                    for (uint16_t softmax_idx = 0; softmax_idx < softmax_length; softmax_idx++) {
                        uint32_t softmax_value_offset = softmax_vector_base_offset + softmax_idx * X->dims[3];
                        for (uint16_t intra_tile_idx3 = 0; intra_tile_idx3 < idx3_tile_size; intra_tile_idx3++) {
                            op_buffer[intra_tile_idx3] = lea_buffer[intra_tile_idx3 * softmax_length + softmax_idx];
                        }
                        my_memcpy_to_param(output, softmax_value_offset, op_buffer, idx3_tile_size * sizeof(int16_t), /*timer_delay=*/0, /*is_linear=*/false);
                    }
                    idx3 += idx3_tile_size;
#if HAWAII
                    write_softmax_loop_indices(idx0, idx1, idx3);
#endif
                }
                idx3 = 0;
                idx1++;
            }
            idx1 = 0;
            idx0++;
        }
    } else {
        MY_ASSERT(false);
    }

    dump_params_debug(model, output, node->output_name);

#if HAWAII
    reset_softmax_loop_indices();
#endif
}
