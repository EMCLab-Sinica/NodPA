#include <cstddef>
#include <cstdint>
#include <cinttypes>
#include "cnn_common.h"
#include "counters.h"
#include "data.h"
#include "layer-defs.h"
#include "op_utils.h"
#include "my_debug.h"
#include "intermittent-cnn.h"
#include "my_dsplib.h"
#include "platform.h"

#define RESHAPE_AUTO_DIM static_cast<uint16_t>(-1)

const uint8_t RELU_TILE_SIZE = 16;
static_assert(RELU_TILE_SIZE % BATCH_SIZE == 0, "Incorrect tile size for ReLU");

void alloc_relu(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node*, CurNodeFlags*, const NodeFlags*) {
}

void handle_relu(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node* node, CurNodeFlags*, const NodeFlags*) {
    my_printf_debug("ReLu!" NEWLINE);

    const ParameterInfo *X = input[0];

    uint32_t data_len = X->params_len / 2;

    uint16_t output_offset = 0;
#if INTERMITTENT
    start_cpu_counter(offsetof(Counters, progress_seeking));
    uint32_t first_unfinished_value_offset = batch_start(job_index_to_offset(output, run_recovery(model, output)));
    output_offset += first_unfinished_value_offset;

#if INDIRECT_RECOVERY
    start_cpu_counter(offsetof(Counters, state_query));
    uint16_t next_output_turning_point;
    int16_t offset;
    uint8_t output_turning_point_idx;
    SlotInfo *output_slot_info;
    find_initial_state_bit(&offset, &output_turning_point_idx, &next_output_turning_point, &output_slot_info,
                           first_unfinished_value_offset, model, output);
    offset = -offset;
    stop_cpu_counter();
#endif
    stop_cpu_counter();
#endif

    int16_t vals[32];
    uint16_t i = output_offset;
#if JAPARI
    start_cpu_counter(offsetof(Counters, embedding));
    const uint8_t real_relu_tile_size = extend_for_footprints(RELU_TILE_SIZE);
    stop_cpu_counter();
#else
    const uint8_t real_relu_tile_size = RELU_TILE_SIZE;
#endif
    for (; i < data_len; i += real_relu_tile_size) {
        uint8_t cur_tile_size = MIN_VAL(real_relu_tile_size, data_len - i);
        my_memcpy_from_param(model, vals, X, output_offset, cur_tile_size*sizeof(int16_t));
#if JAPARI && ENABLE_COUNTERS
        add_counter(offsetof(Counters, data_loading), (cur_tile_size/2)*(4*8));
#endif

#if STATEFUL
        start_cpu_counter(offsetof(Counters, stripping));
        for (uint8_t j = 0; j < cur_tile_size; j++) {
            if (offset_has_state(output_offset+j)) {
                strip_state(&vals[j]);
            }
            vals[j] *= 2;
        }
        stop_cpu_counter();
#endif

        for (uint8_t j = 0; j < cur_tile_size; j++) {
            vals[j] = MAX_VAL(vals[j], 0);
        }

#if INDIRECT_RECOVERY
        start_cpu_counter(offsetof(Counters, embedding));
#if STATEFUL
        const uint8_t embedding_shift = BATCH_SIZE;
#else
        const uint8_t embedding_shift = BATCH_SIZE + 1;
#endif
        for (uint8_t j = 0; j < cur_tile_size; j += embedding_shift) {
            uint8_t tile_last = j + embedding_shift - 1;
            start_cpu_counter(offsetof(Counters, state_query));
            check_next_turning_point(offset, output_turning_point_idx, next_output_turning_point, output_slot_info, output_offset + tile_last);
            stop_cpu_counter();
#if STATEFUL
            start_cpu_counter(offsetof(Counters, embedding));
            for (uint8_t k = j; k < tile_last; k++) {
                vals[k] /= 2;
            }
            vals[tile_last] = vals[tile_last] / 2 + offset;
            stop_cpu_counter();
#else
            vals[tile_last] = (offset > 0 ? 1 : -1);
#endif
        }
        stop_cpu_counter();
#endif

#if MY_DEBUG >= MY_DEBUG_VERBOSE
        my_printf_debug("output_offset=[% 6d, % 6d), output_val=", output_offset, output_offset+cur_tile_size);
        for (uint8_t j = 0; j < cur_tile_size; j++) {
            my_printf_debug("% 6d", vals[j]);
            if (j != cur_tile_size - 1) {
                my_printf_debug(", ");
            }
        }
        my_printf_debug(NEWLINE);
#endif

        my_memcpy_to_param(output, output_offset, vals, cur_tile_size*sizeof(int16_t), 0, false);
        output_offset += cur_tile_size;
#if HAWAII
        for (int8_t to_record = cur_tile_size; to_record > 0; to_record -= BATCH_SIZE) {
            write_hawaii_layer_footprint(model->layer_idx, BATCH_SIZE);
        }
#endif
    }

#if INDIRECT_RECOVERY
    start_cpu_counter(offsetof(Counters, table_updates));
    flip_state_bit(model, output);
    stop_cpu_counter();
#endif

    my_printf_debug("handle_relu output" NEWLINE);
    dump_params_nhwc_debug(model, output, node->output_name, "Relu");
}

void handle_reshape(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node* node, CurNodeFlags*, const NodeFlags*) {
    my_printf_debug("Reshape!" NEWLINE);

    const ParameterInfo *data = input[0], *shape = input[1];
    /*
     * At most one dimension of the new shape can be -1. In this case, the
     * value is inferred from the size of the tensor and the remaining
     * dimensions.
     *
     * A dimension could also be 0, in which case the actual dimension value
     * is unchanged (i.e. taken from the input tensor).
     * */
    for (uint8_t i = 0; i < 4 && i < shape->dims[0]; i++) {
        output->dims[i] = get_int64_param(shape, i);
        if (!output->dims[i]) {
            output->dims[i] = data->dims[i];
        }
    }
    for (uint8_t i = shape->dims[0]; i < 4; i++) {
        output->dims[i] = 0;
    }
    uint16_t inferred_dim = output->params_len / sizeof(int16_t);
    int8_t auto_idx = -1;
#if JAPARI
    start_cpu_counter(offsetof(Counters, embedding));
    uint8_t last_dim_idx;
    for (uint8_t i = 0; i < 4; i++) {
        if (output->dims[i]) {
            last_dim_idx = i;
        }
    }
    stop_cpu_counter();
#endif
    for (uint8_t i = 0; i < 4; i++) {
        if (output->dims[i] != RESHAPE_AUTO_DIM && output->dims[i] != 0) {
#if JAPARI
            if (i == last_dim_idx && data->slot != SLOT_TEST_SET) {
                inferred_dim /= extend_for_footprints(output->dims[i]);
            } else
#endif
            {
                inferred_dim /= output->dims[i];
            }
        } else if (output->dims[i] == RESHAPE_AUTO_DIM) {
            auto_idx = i;
        }
    }
    if (auto_idx != -1) {
        output->dims[auto_idx] = inferred_dim;
    }

    dump_params_debug(model, output, node->output_name, "Reshape");
}

void handle_squeeze(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node* node, CurNodeFlags* node_flags, const NodeFlags*) {
    my_printf_debug("Squeeze!" NEWLINE);

    uint8_t axes = node_flags->squeeze.axes;
    // If axes is not provided, all the single dimensions will be removed from the shape.
    // https://github.com/onnx/onnx/blob/master/docs/Operators.md#squeeze
    uint8_t j = 0;
    if (axes == 0) {
        for (uint8_t i = 0; i < 4; i++) {
            if (input[0]->dims[i] != 1) {
                output->dims[j] = input[0]->dims[i];
                j++;
            }
        }
    } else {
        for (uint8_t i = 0; i < 4; i++) {
            if (axes & (1 << i)) {
#if !JAPARI
                MY_ASSERT(input[0]->dims[i] == 1);
#endif
            } else {
                output->dims[j] = input[0]->dims[i];
                j++;
            }
        }
    }
    for (; j < 4; j++) {
        output->dims[j] = 0;
    }
}

void handle_unsqueeze(Model* model, const ParameterInfo* input[], ParameterInfo* output, const Node* node, CurNodeFlags* node_flags, const NodeFlags*) {
    my_printf_debug("Unsqueeze!" NEWLINE);
    uint8_t axes = node_flags->squeeze.axes;
    uint8_t input_dim_offset = 0, output_dim_offset = 0;
    for (uint8_t i = 0; i < 4; i++) {
        if (axes & (1 << i)) {
            output->dims[output_dim_offset] = 1;
            output_dim_offset++;
        } else {
            output->dims[output_dim_offset] = input[0]->dims[input_dim_offset];
            input_dim_offset++;
            output_dim_offset++;
        }
    }

    dump_params_debug(model, output, node->output_name, "Unsqueeze");
}

void alloc_concat(Model* model, const ParameterInfo *input[], ParameterInfo* output, const Node* node, CurNodeFlags* node_flags, const NodeFlags*) {
    int8_t axis = node_flags->concat.axis;

    output->dims[axis] = 0;
    for (uint8_t input_idx = 0; input_idx < node->inputs_len; input_idx++) {
        const ParameterInfo* inp = input[input_idx];
        MY_ASSERT(inp->dims[axis] <= LEA_BUFFER_SIZE);
#if JAPARI
        // Only support simple cases for now
        MY_ASSERT(inp->dims[axis] % (BATCH_SIZE + 1) == 0);
#elif STATEFUL
        MY_ASSERT(inp->dims[axis] % BATCH_SIZE == 0);
#endif
        output->dims[axis] += inp->dims[axis];
        output->scale = (inp->scale > output->scale) ? inp->scale : output->scale;
    }

    recalculate_params_len(output);
}

static void handle_concat_channels(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node* node, CurNodeFlags*, const NodeFlags*) {
    uint32_t output_offset = 0;
    uint16_t hw = 0;
#if INTERMITTENT
    start_cpu_counter(offsetof(Counters, progress_seeking));
    uint32_t first_unfinished_job_idx = run_recovery(model, output);
    output_offset = batch_start(job_index_to_offset(output, first_unfinished_job_idx));
    hw = output_offset / output->dims[1];

#if INDIRECT_RECOVERY
    start_cpu_counter(offsetof(Counters, state_query));
    uint16_t next_output_turning_point;
    int16_t offset;
    uint8_t output_turning_point_idx;
    SlotInfo *output_slot_info;
    find_initial_state_bit(&offset, &output_turning_point_idx, &next_output_turning_point, &output_slot_info,
                           output_offset, model, output);
    stop_cpu_counter();
#endif
    stop_cpu_counter();

#endif

    uint16_t already_copied = output_offset - hw * output->dims[1];
    for (; hw < output->dims[2] * output->dims[3]; hw++) {
        for (uint8_t input_idx = 0; input_idx < node->inputs_len; input_idx++) {
            const ParameterInfo* inp = input[input_idx];
            const int16_t input_channels = inp->dims[1];
            if (already_copied >= input_channels) {
                already_copied -= input_channels;
                continue;
            }
            uint16_t to_copy = input_channels - already_copied;
#if INDIRECT_RECOVERY
            start_cpu_counter(offsetof(Counters, state_query));
            fill_state_offsets(output_offset, to_copy, &offset, &output_turning_point_idx, &next_output_turning_point, output_slot_info);
            stop_cpu_counter();
#endif
            my_memcpy_from_param(model, lea_buffer, inp, hw * input_channels + already_copied, to_copy * sizeof(int16_t));
            already_copied = 0;
#if STATEFUL
            for (uint16_t idx = BATCH_SIZE - 1; idx < to_copy; idx += BATCH_SIZE) {
                strip_state(lea_buffer + idx);
            }
#endif
            if (inp->scale != output->scale) {
                int16_t scaleFract;
                uint8_t shift;
                float_to_scale_params(&scaleFract, &shift, inp->scale/output->scale);
                my_scale_q15(lea_buffer, scaleFract, shift, lea_buffer, to_copy * sizeof(int16_t));
            }

#if INDIRECT_RECOVERY
            start_cpu_counter(offsetof(Counters, embedding));
            update_states(lea_buffer, to_copy, true);
            stop_cpu_counter();
#endif

            my_memcpy_to_param(output, output_offset, lea_buffer, to_copy * sizeof(int16_t), 0, true); // XXX: is Concat linear layer?
            my_printf_debug("Copied %u values to [%d, %d)" NEWLINE, to_copy, output_offset, output_offset + to_copy);
            output_offset += to_copy;
#if HAWAII
            write_hawaii_layer_footprint(model->layer_idx, to_copy);
#endif
        }
    }

    dump_params_nhwc_debug(model, output, node->output_name, "Concat");

#if INDIRECT_RECOVERY
    start_cpu_counter(offsetof(Counters, table_updates));
    flip_state_bit(model, output);
    stop_cpu_counter();
#endif
}

static void handle_concat_batch(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node* node, CurNodeFlags* node_flags, const NodeFlags*) {
    uint32_t part_len = input[0]->params_len;

    uint8_t input_idx = 0;
    uint32_t copy_offset = 0;
#if INTERMITTENT
    uint32_t first_unfinished_job_idx = run_recovery(model, output);
    uint32_t data_offset = batch_start(job_index_to_offset(output, first_unfinished_job_idx));
    input_idx = data_offset / part_len;
    copy_offset = data_offset % part_len;
#endif

    for (; input_idx < node->inputs_len; input_idx++) {
        const uint32_t copy_len = MIN_VAL(LIMIT_DMA_SIZE(LEA_BUFFER_SIZE), 32);
        for (; copy_offset < part_len; copy_offset += copy_len) {
            const uint32_t cur_copy_len = MIN_VAL(copy_len, part_len - copy_offset);
            my_memcpy_from_param(model, lea_buffer, input[input_idx], copy_offset/sizeof(int16_t), cur_copy_len);
            my_memcpy_to_param(output, (input_idx * part_len + copy_offset)/sizeof(int16_t), lea_buffer, cur_copy_len, /*timer_delay=*/0, /*is_linear=*/false);
#if HAWAII
            write_hawaii_layer_footprint(model->layer_idx, cur_copy_len/sizeof(int16_t));
#endif
        }
        copy_offset = 0;
    }

    dump_params_debug(model, output, node->output_name, "Concat");
}

void handle_concat(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node* node, CurNodeFlags* node_flags, const NodeFlags* orig_node_flags) {
    my_printf_debug("Concat!" NEWLINE);

    // Only batch or channel concatenation is supported for now
    uint8_t axis = node_flags->concat.axis;
    if (axis == 0) {
        handle_concat_batch(model, input, output, node, node_flags, orig_node_flags);
    } else if (axis == 1) {
        handle_concat_channels(model, input, output, node, node_flags, orig_node_flags);
    } else {
        MY_ASSERT(false);
    }
}

void alloc_transpose(struct Model *model, const struct ParameterInfo **input, struct ParameterInfo *output, const struct Node *node, CurNodeFlags* node_flags, const NodeFlags*) {
    const ParameterInfo *X = input[0];

    const int8_t* perm = node_flags->transpose.perm;
    for (uint8_t dim_idx = 0; dim_idx < 4; dim_idx++) {
        if (perm[dim_idx] < 0) {
            break;
        }
        output->dims[dim_idx] = X->dims[perm[dim_idx]];
    }
}

void handle_transpose(Model* model, const ParameterInfo *input[], ParameterInfo *output, const Node* node, CurNodeFlags* node_flags, const NodeFlags*) {
    my_printf_debug("Transpose!" NEWLINE);

    const ParameterInfo *X = input[0];

    const int8_t* inverse_perm = node_flags->transpose.inverse_perm;

    uint32_t data_offset = 0;
#if INTERMITTENT
    start_cpu_counter(offsetof(Counters, progress_seeking));
    uint32_t first_unfinished_job_idx = run_recovery(model, output);
    data_offset = batch_start(job_index_to_offset(output, first_unfinished_job_idx));
    stop_cpu_counter();

#if INDIRECT_RECOVERY
    start_cpu_counter(offsetof(Counters, state_query));
    uint16_t next_output_turning_point;
    int16_t offset;
    uint8_t output_turning_point_idx;
    SlotInfo *output_slot_info;
    find_initial_state_bit(&offset, &output_turning_point_idx, &next_output_turning_point, &output_slot_info,
                           data_offset, model, output);
    stop_cpu_counter();
#endif

#endif

    uint16_t input_indices[4], output_indices[4];

    uint8_t N_dims = count_dims(X);

    if (N_dims == 4) {
        output_indices[3] = data_offset % output->dims[3];
        data_offset /= output->dims[3];
    }
    output_indices[2] = data_offset % output->dims[2];
    data_offset /= output->dims[2];
    output_indices[1] = data_offset % output->dims[1];
    data_offset /= output->dims[1];
    output_indices[0] = data_offset;
    if (N_dims == 4) {
        my_printf_debug("output_indices: [%d, %d, %d, %d]" NEWLINE, output_indices[0], output_indices[1], output_indices[2], output_indices[3]);
    } else if (N_dims == 3) {
        my_printf_debug("output_indices: [%d, %d, %d]" NEWLINE, output_indices[0], output_indices[1], output_indices[2]);
    }

    if (N_dims == 4) {
        const uint16_t vector_size = output->dims[3];

        for (; output_indices[0] < output->dims[0]; output_indices[0]++) {
            for (; output_indices[1] < output->dims[1]; output_indices[1]++) {
                for (; output_indices[2] < output->dims[2]; output_indices[2]++) {
#if INDIRECT_RECOVERY
                    start_cpu_counter(offsetof(Counters, state_query));
                    fill_state_offsets(data_offset, vector_size, &offset, &output_turning_point_idx, &next_output_turning_point, output_slot_info);
                    stop_cpu_counter();
#endif

                    for (; output_indices[3] < vector_size; output_indices[3]++) {
                        for (uint8_t dim_idx = 0; dim_idx < 4; dim_idx++) {
                            input_indices[dim_idx] = output_indices[inverse_perm[dim_idx]];
                        }
                        uint32_t input_offset = input_indices[0] * X->dims[1] * X->dims[2] * X->dims[3]+
                                                input_indices[1] * X->dims[2] * X->dims[3] +
                                                input_indices[2] * X->dims[3] +
                                                input_indices[3];
                        uint32_t output_offset = output_indices[0] * output->dims[1] * output->dims[2] * output->dims[3] +
                                                 output_indices[1] * output->dims[2] * output->dims[3] +
                                                 output_indices[2] * output->dims[3] +
                                                 output_indices[3];
                        int16_t val = get_q15_param(model, X, input_offset);
                        my_printf_debug("input_offset=%" PRIu32 " val=%d" NEWLINE, input_offset, val);

#if STATEFUL
                        strip_state(&val);
#endif

#if INDIRECT_RECOVERY
                        start_cpu_counter(offsetof(Counters, embedding));
                        val += state_offsets[output_indices[3]];
                        stop_cpu_counter();
#endif

                        my_printf_debug("output_offset=%" PRIu32 " val=%d" NEWLINE, output_offset, val);
                        put_q15_param(output, output_offset, val, /*is_linear=*/false);
#if HAWAII
                        write_hawaii_layer_footprint(model->layer_idx, /*n_jobs=*/1);
#endif
                    }
                    output_indices[3] = 0;
                }
                output_indices[2] = 0;
            }
            output_indices[1] = 0;
        }
    } else if (N_dims == 3) {
        const uint16_t vector_size = output->dims[2];

        for (; output_indices[0] < output->dims[0]; output_indices[0]++) {
            for (; output_indices[1] < output->dims[1]; output_indices[1]++) {
#if INDIRECT_RECOVERY
                start_cpu_counter(offsetof(Counters, state_query));
                fill_state_offsets(data_offset, vector_size, &offset, &output_turning_point_idx, &next_output_turning_point, output_slot_info);
                stop_cpu_counter();
#endif

                for (; output_indices[2] < vector_size; output_indices[2]++) {
                    for (uint8_t dim_idx = 0; dim_idx < 3; dim_idx++) {
                        input_indices[dim_idx] = output_indices[inverse_perm[dim_idx]];
                    }
                    uint32_t input_offset = input_indices[0] * X->dims[1] * X->dims[2] +
                                            input_indices[1] * X->dims[2] +
                                            input_indices[2];
                    uint32_t output_offset = output_indices[0] * output->dims[1] * output->dims[2] +
                                             output_indices[1] * output->dims[2] +
                                             output_indices[2];
                    int16_t val = get_q15_param(model, X, input_offset);
                    my_printf_debug("input_offset=%" PRIu32 " val=%d" NEWLINE, input_offset, val);

#if STATEFUL
                    strip_state(&val);
#endif

#if INDIRECT_RECOVERY
                    start_cpu_counter(offsetof(Counters, embedding));
                    val += state_offsets[output_indices[2]];
                    stop_cpu_counter();
#endif

                    my_printf_debug("output_offset=%" PRIu32 " val=%d" NEWLINE, output_offset, val);
                    put_q15_param(output, output_offset, val, /*is_linear=*/false);
#if HAWAII
                    write_hawaii_layer_footprint(model->layer_idx, /*n_jobs=*/1);
#endif
                }
                output_indices[2] = 0;
            }
            output_indices[1] = 0;
        }
    }

#if INDIRECT_RECOVERY
    start_cpu_counter(offsetof(Counters, table_updates));
    flip_state_bit(model, output);
    stop_cpu_counter();
#endif

    dump_params_debug(model, output, node->output_name, "Transpose");
}

void alloc_add(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node *node, CurNodeFlags*, const NodeFlags*) {
}

void handle_add(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node *node, CurNodeFlags* node_flags, const NodeFlags*) {
    my_printf_debug("Add!" NEWLINE);

    const ParameterInfo *X = input[0], *Y = input[1];

    uint32_t data_offset = 0;
#if INTERMITTENT
    start_cpu_counter(offsetof(Counters, progress_seeking));
    uint32_t first_unfinished_job_idx = run_recovery(model, output);
    data_offset = batch_start(job_index_to_offset(output, first_unfinished_job_idx));

    fix_first_unfinished_value_offset(model, &data_offset);

#if INDIRECT_RECOVERY
    start_cpu_counter(offsetof(Counters, state_query));
    uint16_t next_output_turning_point;
    int16_t output_offset;
    uint8_t output_turning_point_idx;
    SlotInfo* output_slot_info;
    find_initial_state_bit(&output_offset, &output_turning_point_idx, &next_output_turning_point, &output_slot_info,
                           data_offset, model, output);
    stop_cpu_counter();
#endif
    stop_cpu_counter();
#endif

    uint16_t original_buffer_size, buffer_size;
    if (X->param_flags & CHANNEL_LAST) {
        buffer_size = Y->dims[1];
    } else {
        buffer_size = Y->dims[count_dims(Y) - 1];
    }
    original_buffer_size = buffer_size;
#if JAPARI
    buffer_size = extend_for_footprints(buffer_size);
#endif

    int16_t *buffer_x = lea_buffer,
            *buffer_y = buffer_x + buffer_size;
    my_memcpy_from_param(model, buffer_y, Y, 0, original_buffer_size * sizeof(int16_t));
#if JAPARI
    start_cpu_counter(offsetof(Counters, embedding));
    move_weights(buffer_y, false, buffer_size, original_buffer_size);
    stop_cpu_counter();
#endif
    my_printf_debug("Y" NEWLINE);
    dump_matrix_debug(buffer_y, buffer_size, ValueInfo(Y), false);

    int16_t scaleFract;
    uint8_t shift;
    float_to_scale_params(&scaleFract, &shift, Y->scale/X->scale);
    my_scale_q15(buffer_y, scaleFract, shift, buffer_y, buffer_size);

    uint16_t idx = data_offset / buffer_size;
    uint16_t cur_buffer_size = buffer_size - (data_offset - idx * buffer_size);
    for (; idx < X->dims[node_flags->add.weights_broadcasted_dim]; idx++) {
        my_printf_debug("data_offset=%d" NEWLINE, data_offset);
        my_memcpy_from_param(model, buffer_x, X, data_offset, cur_buffer_size * sizeof(int16_t));
#if STATEFUL
        my_printf_debug("Before strip states" NEWLINE);
        dump_matrix_debug(buffer_x, cur_buffer_size, ValueInfo(output), false);

        for (uint16_t val_idx = BATCH_SIZE - 1; val_idx < cur_buffer_size; val_idx += BATCH_SIZE) {
            strip_state(buffer_x + val_idx);
        }

        my_printf_debug("After strip states" NEWLINE);
        dump_matrix_debug(buffer_x, cur_buffer_size, ValueInfo(output), false);
#endif

        my_add_q15(buffer_x, buffer_y + (buffer_size - cur_buffer_size), buffer_x, cur_buffer_size);
        my_printf_debug("After add" NEWLINE);
        dump_matrix_debug(buffer_x, cur_buffer_size, ValueInfo(output), false);

#if INDIRECT_RECOVERY
        start_cpu_counter(offsetof(Counters, state_query));
        fill_state_offsets(data_offset, cur_buffer_size, &output_offset, &output_turning_point_idx, &next_output_turning_point, output_slot_info);
        stop_cpu_counter();
        start_cpu_counter(offsetof(Counters, embedding));
        update_states(buffer_x, cur_buffer_size, true);
        stop_cpu_counter();
        my_printf_debug("After embedding states" NEWLINE);
        dump_matrix_debug(buffer_x, cur_buffer_size, ValueInfo(output), true);
#endif

        my_memcpy_to_param(output, data_offset, buffer_x, cur_buffer_size * sizeof(int16_t), 0, true);
        data_offset += cur_buffer_size;
#if HAWAII
        write_hawaii_layer_footprint(model->layer_idx, cur_buffer_size/BATCH_SIZE*BATCH_SIZE);
#endif
        cur_buffer_size = buffer_size;
    }

#if INDIRECT_RECOVERY
    start_cpu_counter(offsetof(Counters, table_updates));
    flip_state_bit(model, output);
    stop_cpu_counter();
#endif

    if (X->param_flags & CHANNEL_LAST) {
        dump_params_nhwc_debug(model, output, node->output_name, "Add");
    } else {
        dump_params_debug(model, output, node->output_name, "Add");
    }
}
