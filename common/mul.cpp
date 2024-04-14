#include <cstdint>
#include "cnn_common.h"
#include "counters.h"
#include "data.h"
#include "intermittent-cnn.h"
#include "layer-defs.h"
#include "my_debug.h"
#include "my_dsplib.h"
#include "op_utils.h"
#include "platform.h"

void alloc_mul(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node *node, CurNodeFlags*, const NodeFlags*) {
    const ParameterInfo *X = input[0], *Y = input[1];

    output->slot = get_next_slot(model, input[0]);
    output->scale = X->scale * Y->scale;
}

void handle_mul(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node *node, CurNodeFlags*, const NodeFlags*) {
    my_printf_debug("Mul!" NEWLINE);

    const ParameterInfo *X = input[0], *Y = input[1];

    uint32_t data_offset = 0;

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

    uint16_t buffer_size = 256;
    int16_t *buffer_x = lea_buffer,
            *buffer_y = buffer_x + buffer_size;
    my_memcpy_from_param(model, buffer_y, Y, 0, sizeof(int16_t));

    int16_t scaleFract;
    uint8_t shift;
    float_to_scale_params(&scaleFract, &shift, q15_to_float(*buffer_y, ValueInfo(Y), /*p_use_prefix=*/nullptr, /*has_state=*/false));

    for (; data_offset < output->params_len / sizeof(int16_t); data_offset += buffer_size) {
        uint16_t cur_buffer_size = MIN_VAL(output->params_len / sizeof(int16_t) - data_offset, buffer_size);
        my_printf_debug("data_offset=%d" NEWLINE, data_offset);
        my_memcpy_from_param(model, buffer_x, X, data_offset, cur_buffer_size * sizeof(int16_t));

#if STATEFUL
        my_printf_debug("Before strip states" NEWLINE);
        dump_matrix_debug(buffer_x, cur_buffer_size, ValueInfo(output), false);

        for (uint16_t val_idx = BATCH_SIZE - 1; val_idx < cur_buffer_size; val_idx += BATCH_SIZE) {
            strip_state(buffer_x + val_idx);
        }

#endif
        my_printf_debug("Before mul" NEWLINE);
        dump_matrix_debug(buffer_x, cur_buffer_size, ValueInfo(output), false);

        my_scale_q15(buffer_x, scaleFract, shift, buffer_x, buffer_size);
        my_printf_debug("After mul" NEWLINE);
        dump_matrix_debug(buffer_x, cur_buffer_size, ValueInfo(output), false);

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

        my_memcpy_to_param(output, data_offset, buffer_x, cur_buffer_size * sizeof(int16_t), 0, true);

#if HAWAII
        write_hawaii_layer_footprint(model->layer_idx, cur_buffer_size);
#endif
    }

#if INDIRECT_RECOVERY
    start_cpu_counter(offsetof(Counters, table_updates));
    flip_state_bit(model, output);
    stop_cpu_counter();
#endif

    dump_params_debug(model, output, node->output_name);
}
