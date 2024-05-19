#include "cnn_common.h"
#include "counters.h"
#include "data.h"
#include "intermittent-cnn.h"
#include "layer-defs.h"
#include "my_debug.h"
#include "my_dsplib.h"
#include "op_utils.h"
#include "platform.h"
#include <cstdint>

typedef void (*vector_op)(const ParameterInfo* X, const ParameterInfo* Y, int16_t* buffer_x, int16_t* buffer_y, uint16_t buffer_size, bool cached_y);

static struct {
    uint8_t X_num_dims;
    uint8_t Y_num_dims;
    uint8_t output_num_dims;
    bool channel_last;
} binary_op_params;

static void alloc_broadcasted_binary_op(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node *node, CurNodeFlags*, const NodeFlags*) {
    const ParameterInfo *X = input[0], *Y = input[1];

    // Broadcast dimensions
    // https://numpy.org/doc/stable/user/basics.broadcasting.html#general-broadcasting-rules
    binary_op_params.X_num_dims = count_dims(X);
    binary_op_params.Y_num_dims = count_dims(Y);
    binary_op_params.output_num_dims = MAX_VAL(binary_op_params.X_num_dims, binary_op_params.Y_num_dims);
    binary_op_params.channel_last = ((X->param_flags | Y->param_flags) & CHANNEL_LAST) != 0;

    uint8_t X_dim_offset = binary_op_params.output_num_dims - binary_op_params.X_num_dims,
            Y_dim_offset = binary_op_params.output_num_dims - binary_op_params.Y_num_dims;
    for (int8_t dim_idx = binary_op_params.output_num_dims - 1; dim_idx >= 0; dim_idx--) {
        if (dim_idx >= X_dim_offset && dim_idx >= Y_dim_offset) {
            uint16_t X_dim = X->dims[dim_idx - X_dim_offset],
                     Y_dim = Y->dims[dim_idx - Y_dim_offset];

#if JAPARI
            if (dim_idx == 1) {
                MY_ASSERT((X_dim == 1) || (Y_dim == 1) || (X_dim == Y_dim * 2), "Invalid broadcasting: incompatible dimensions");
            } else
#endif
            {
                MY_ASSERT((X_dim == 1) || (Y_dim == 1) || (X_dim == Y_dim), "Invalid broadcasting: incompatible dimensions");
            }

            output->dims[dim_idx] = MAX_VAL(X_dim, Y_dim);
        } else if (dim_idx >= X_dim_offset) {
            uint16_t X_dim = X->dims[dim_idx - X_dim_offset];
            output->dims[dim_idx] = X_dim;
        } else {
            uint16_t Y_dim = Y->dims[dim_idx - Y_dim_offset];
            output->dims[dim_idx] = Y_dim;
        }
    }

    recalculate_params_len(output);

    if (binary_op_params.channel_last) {
        output->param_flags |= CHANNEL_LAST;
    }
}

static uint32_t get_broadcast_index(const uint16_t input_dims[], uint8_t input_num_dims, const uint16_t output_indices[], const uint32_t strides[]) {
    uint32_t input_value_offset = 0;
    for (int8_t dim_idx = input_num_dims - 1; dim_idx >= 0; dim_idx--) {
        if (output_indices[dim_idx] < input_dims[dim_idx]) {
            input_value_offset += output_indices[dim_idx] * strides[dim_idx];
        }
    }
    return input_value_offset;
}

static void map_dims_channel_last(const uint16_t dims[], uint16_t mapped_dims[], uint8_t num_dims) {
    mapped_dims[0] = dims[0];
    mapped_dims[binary_op_params.X_num_dims - 1] = dims[1];
    for (uint8_t dim_idx = 1; dim_idx < binary_op_params.X_num_dims - 1; dim_idx++) {
        mapped_dims[dim_idx] = dims[dim_idx+1];
    }
}

static void calculate_strides(uint32_t strides[], const uint16_t dims[], uint16_t num_dims) {
    strides[num_dims - 1] = 1;
    for (int8_t dim_idx = binary_op_params.X_num_dims - 2; dim_idx >= 0; dim_idx--) {
        strides[dim_idx] = strides[dim_idx+1] * dims[dim_idx+1];
    }
}

void handle_broadcastd_binary_op(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node *node, CurNodeFlags*, const NodeFlags*,
                                 vector_op broadcasted_vector_op, vector_op non_broadcasted_vector_op) {
    const ParameterInfo *X = input[0], *Y = input[1];

    uint32_t data_offset = 0;
    uint16_t output_indices[MAX_NUM_DIMS];

#if INTERMITTENT
    start_cpu_counter(offsetof(Counters, progress_seeking));
    uint32_t first_unfinished_job_idx = run_recovery(model, output);
    data_offset = batch_start(job_index_to_offset(output, first_unfinished_job_idx));

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

    stop_cpu_counter();
#endif

    // Map dimensions for CHANNEL_LAST
    const uint16_t *X_dims, *Y_dims, *output_dims;
    uint16_t X_mapped_dims[MAX_NUM_DIMS], Y_mapped_dims[MAX_NUM_DIMS], output_mapped_dims[MAX_NUM_DIMS];
    if (binary_op_params.channel_last) {
        map_dims_channel_last(X->dims, X_mapped_dims, binary_op_params.X_num_dims);
        map_dims_channel_last(Y->dims, Y_mapped_dims, binary_op_params.Y_num_dims);
        map_dims_channel_last(output->dims, output_mapped_dims, binary_op_params.output_num_dims);

        X_dims = X_mapped_dims;
        Y_dims = Y_mapped_dims;
        output_dims = output_mapped_dims;
    } else {
        X_dims = X->dims;
        Y_dims = Y->dims;
        output_dims = output->dims;
    }

    uint32_t remaining_data_offset = data_offset;
    for (int8_t dim_idx = binary_op_params.output_num_dims - 1; dim_idx >= 0; dim_idx--) {
        output_indices[dim_idx] = remaining_data_offset % output_dims[dim_idx];
        remaining_data_offset /= output_dims[dim_idx];
    }

    uint16_t* output_indices_without_footprints;
#if JAPARI
    uint16_t output_indices_without_footprints_arr[MAX_NUM_DIMS];
    if (output_indices[binary_op_params.output_num_dims - 1] != 0) {
        my_memcpy(output_indices_without_footprints_arr, output_indices, (binary_op_params.output_num_dims - 1) * sizeof(uint16_t));
        output_indices_without_footprints_arr[binary_op_params.output_num_dims - 1] = offset_without_footprints(output_indices[binary_op_params.output_num_dims - 1]);
        output_indices_without_footprints = output_indices_without_footprints_arr;
    } else
#endif
    {
        output_indices_without_footprints = output_indices;
    }

    uint32_t X_strides[MAX_NUM_DIMS], Y_strides[MAX_NUM_DIMS];
    calculate_strides(X_strides, X_dims, binary_op_params.X_num_dims);
    calculate_strides(Y_strides, Y_dims, binary_op_params.Y_num_dims);

    uint16_t X_last_dim = X_dims[binary_op_params.X_num_dims - 1],
             Y_last_dim = Y_dims[binary_op_params.Y_num_dims - 1];
    uint16_t buffer_size = MAX_VAL(X_last_dim, Y_last_dim);
    int16_t *buffer_x = lea_buffer,
            *buffer_y = buffer_x + buffer_size;

#if JAPARI
    uint16_t buffer_size_without_footprints = buffer_size;
    buffer_size_without_footprints = offset_without_footprints(buffer_size);
#endif

    uint32_t cached_Y_value_offset = UINT32_MAX;

    while (data_offset < output->params_len / sizeof(int16_t)) {
        uint16_t cur_buffer_size = buffer_size - (data_offset % buffer_size);

        uint16_t cur_buffer_size_without_footprints = cur_buffer_size;
#if JAPARI
        cur_buffer_size_without_footprints = offset_without_footprints(cur_buffer_size);
#endif

        uint32_t X_value_offset = get_broadcast_index(X_dims, binary_op_params.X_num_dims, output_indices, X_strides),
                 Y_value_offset = get_broadcast_index(Y_dims, binary_op_params.Y_num_dims, output_indices_without_footprints, Y_strides);

        bool cached_y = (cached_Y_value_offset == Y_value_offset);

        my_printf_debug(NEWLINE "X_value_offset=%d, Y_value_offset=%d, data_offset=%d" NEWLINE,
                        X_value_offset, Y_value_offset, data_offset);
#if MY_DEBUG >= MY_DEBUG_VERBOSE
        my_printf_debug("output_indices=");
        for (uint8_t dim_idx = 0; dim_idx < binary_op_params.output_num_dims; dim_idx++) {
            my_printf_debug("%d ", output_indices[dim_idx]);
        }
        my_printf_debug(NEWLINE);
#endif

        my_memcpy_from_param(model, buffer_x, X, X_value_offset, cur_buffer_size * sizeof(int16_t));

#if STATEFUL
        my_printf_debug("Before strip states" NEWLINE);
        dump_matrix_debug(buffer_x, cur_buffer_size, ValueInfo(output), false);

        for (uint16_t val_idx = BATCH_SIZE - 1; val_idx < cur_buffer_size; val_idx += BATCH_SIZE) {
            strip_state(buffer_x + val_idx);
        }
#endif

        my_printf_debug("Before computation" NEWLINE);
        dump_matrix_debug(buffer_x, cur_buffer_size, ValueInfo(X));

        bool broadcasted_Y;
#if JAPARI
        if (binary_op_params.channel_last) {
            broadcasted_Y = (X_last_dim != Y_last_dim * 2);
        } else
#endif
        {
            broadcasted_Y = (X_last_dim != Y_last_dim);
        }

        if (broadcasted_Y) {
            if (!cached_y) {
                my_memcpy_from_param(model, buffer_y, Y, Y_value_offset, sizeof(int16_t));
                my_printf_debug("Y" NEWLINE);
                dump_matrix_debug(buffer_y, 1, ValueInfo(Y));
            }

            broadcasted_vector_op(X, Y, buffer_x, buffer_y, cur_buffer_size, cached_y);
        } else {
            if (!cached_y) {
                my_memcpy_from_param(model, buffer_y, Y, Y_value_offset, cur_buffer_size_without_footprints * sizeof(int16_t));

#if JAPARI
                start_cpu_counter(offsetof(Counters, embedding));
                move_weights(buffer_y, false, buffer_size, buffer_size_without_footprints);
                stop_cpu_counter();
#endif

                my_printf_debug("Y" NEWLINE);
                dump_matrix_debug(buffer_y, cur_buffer_size, ValueInfo(Y));
            }

            non_broadcasted_vector_op(X, Y, buffer_x, buffer_y, cur_buffer_size, cached_y);
        }

        my_printf_debug("After computation" NEWLINE);
        dump_matrix_debug(buffer_x, cur_buffer_size, ValueInfo(X));

#if INDIRECT_RECOVERY
        start_cpu_counter(offsetof(Counters, state_query));
        fill_state_offsets(data_offset, cur_buffer_size, &offset, &output_turning_point_idx, &next_output_turning_point, output_slot_info);
        stop_cpu_counter();
        start_cpu_counter(offsetof(Counters, embedding));
        update_states(buffer_x, cur_buffer_size, true);
        stop_cpu_counter();
        my_printf_debug("After embedding states" NEWLINE);
        dump_matrix_debug(buffer_x, cur_buffer_size, ValueInfo(output), true);
#endif

        my_memcpy_to_param(output, data_offset, buffer_x, cur_buffer_size * sizeof(int16_t), /*timer_delay=*/0, /*is_linear=*/true);

        // Increment output_indices
        // One vector is processed at a time - reset the last index to 0
        output_indices[binary_op_params.output_num_dims - 1] = 0;
#if JAPARI
        output_indices_without_footprints[binary_op_params.output_num_dims - 1] = 0;
#endif
        // Update remaining dimensions normally
        for (int8_t dim_idx = (binary_op_params.output_num_dims - 1) - 1; dim_idx >= 0; dim_idx--) {
            output_indices[dim_idx]++;
            if (output_indices[dim_idx] < output_dims[dim_idx]) {
                break;
            }
            output_indices[dim_idx] = 0;
        }
#if HAWAII
        write_hawaii_layer_footprint(model->layer_idx, cur_buffer_size);
#endif

        data_offset += cur_buffer_size;

        cached_Y_value_offset = Y_value_offset;
    }

#if INDIRECT_RECOVERY
    start_cpu_counter(offsetof(Counters, table_updates));
    flip_state_bit(model, output);
    stop_cpu_counter();
#endif
}

void alloc_mul(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node *node, CurNodeFlags* node_flags, const NodeFlags* orig_node_flags) {
    const ParameterInfo *X = input[0], *Y = input[1];

    alloc_broadcasted_binary_op(model, input, output, node, node_flags, orig_node_flags);

    output->scale = X->scale * Y->scale;
}

static void broadcasted_vector_mul(const ParameterInfo* X, const ParameterInfo* Y, int16_t* buffer_x, int16_t* buffer_y, uint16_t buffer_size, bool cached_y) {
    static int16_t scaleFract;
    static uint8_t shift;

    if (!cached_y) {
        float_to_scale_params(&scaleFract, &shift, q15_to_float(*buffer_y, ValueInfo(Y), /*p_use_prefix=*/nullptr, /*has_state=*/false));
    }

    my_scale_q15(buffer_x, scaleFract, shift, buffer_x, buffer_size);
}

static void non_broadcasted_vector_mul(const ParameterInfo* X, const ParameterInfo* Y, int16_t* buffer_x, int16_t* buffer_y, uint16_t buffer_size, bool cached_y) {
    my_vector_mult_q15(buffer_x, buffer_y, buffer_x, buffer_size);
}

void handle_mul(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node *node, CurNodeFlags* node_flags, const NodeFlags* orig_node_flags) {
    my_printf_debug("Mul!" NEWLINE);

    handle_broadcastd_binary_op(model, input, output, node, node_flags, orig_node_flags, broadcasted_vector_mul, non_broadcasted_vector_mul);

    if (binary_op_params.channel_last) {
        dump_params_nhwc_debug(model, output, node->output_name, "Mul");
    } else {
        dump_params_debug(model, output, node->output_name, "Mul");
    }
}

void alloc_add(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node *node, CurNodeFlags* node_flags, const NodeFlags* orig_node_flags) {
    alloc_broadcasted_binary_op(model, input, output, node, node_flags, orig_node_flags);
}

static void align_scale(const ParameterInfo* X, const ParameterInfo* Y, int16_t* buffer_y, uint16_t buffer_size, bool cached_y) {
    if (cached_y) {
        return;
    }

    int16_t scaleFract;
    uint8_t shift;
    float_to_scale_params(&scaleFract, &shift, Y->scale/X->scale);
    my_scale_q15(buffer_y, scaleFract, shift, buffer_y, buffer_size);
}

static void broadcasted_vector_add(const ParameterInfo* X, const ParameterInfo* Y, int16_t* buffer_x, int16_t* buffer_y, uint16_t buffer_size, bool cached_y) {
    align_scale(X, Y, buffer_y, buffer_size, cached_y);
    my_offset_q15(buffer_x, *buffer_y, buffer_x, buffer_size);
}

static void non_broadcasted_vector_add(const ParameterInfo* X, const ParameterInfo* Y, int16_t* buffer_x, int16_t* buffer_y, uint16_t buffer_size, bool cached_y) {
    align_scale(X, Y, buffer_y, buffer_size, cached_y);
    my_add_q15(buffer_x, buffer_y, buffer_x, buffer_size);
}

void handle_add(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node *node, CurNodeFlags* node_flags, const NodeFlags* orig_node_flags) {
    my_printf_debug("Add!" NEWLINE);

    handle_broadcastd_binary_op(model, input, output, node, node_flags, orig_node_flags, broadcasted_vector_add, non_broadcasted_vector_add);

    if (binary_op_params.channel_last) {
        dump_params_nhwc_debug(model, output, node->output_name, "Add");
    } else {
        dump_params_debug(model, output, node->output_name, "Add");
    }
}
