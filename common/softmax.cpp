#include <cmath>
#include <cstdint>

#include "DSPLib.h"

#include "cnn_common.h"
#include "counters.h"
#include "data.h"
#include "intermittent-cnn.h"
#include "layer-defs.h"
#include "my_debug.h"
#include "my_dsplib.h"
#include "op_utils.h"
#include "platform.h"

void alloc_softmax(Model* model, const ParameterInfo* input[], ParameterInfo* output, const Node* node, CurNodeFlags* node_flags, const NodeFlags*) {
        output->slot = get_next_slot(model, input[0]);
}

void handle_softmax(Model* model, const ParameterInfo* input[], ParameterInfo* output, const Node* node, CurNodeFlags* node_flags, const NodeFlags*) {
    int axis = node_flags->softmax.axis;

    const ParameterInfo* X = input[0];
    const uint16_t softmax_length = X->dims[axis];

    uint32_t softmax_vector_idx = 0;
    uint32_t softmax_num_vectors = output->params_len / sizeof(int16_t) / softmax_length;
    uint32_t data_offset = 0;
    uint16_t softmax_idx = 0;
#if INTERMITTENT
    start_cpu_counter(offsetof(Counters, progress_seeking));
    uint32_t first_unfinished_job_idx = run_recovery(model, output);
    data_offset = batch_start(job_index_to_offset(output, first_unfinished_job_idx));
    stop_cpu_counter();

    softmax_vector_idx = data_offset / softmax_length;
    softmax_idx = data_offset % softmax_length;
    // needs to start from the row head
    data_offset -= softmax_idx;
#endif

    for (; softmax_vector_idx < softmax_num_vectors; softmax_vector_idx++, data_offset += softmax_length) {
        my_memcpy_from_param(model, lea_buffer, X, data_offset, softmax_length * sizeof(int16_t));

        // avoid exponential overflow and underflow
        // https://www.cnblogs.com/guoyaohua/p/8900683.html
        int16_t max_val = 0;
        uint16_t max_val_idx = 0;
        my_max_q15(lea_buffer, softmax_length, &max_val, &max_val_idx);
        my_offset_q15(lea_buffer, -(max_val), lea_buffer, softmax_length);

        for (; softmax_idx < softmax_length; softmax_idx++) {
            // exponentials
            float val = q15_to_float(lea_buffer[softmax_idx], ValueInfo(output));
            val = std::exp(val);
            int16_t exp_val = _Q15(val / output->scale.toFloat());
            put_q15_param(output, data_offset + softmax_idx, exp_val, /*is_linear=*/false);
#if HAWAII
            write_hawaii_layer_footprint(model->layer_idx, 1);
#endif
        }
        softmax_idx = 0;
    }

    dump_params_debug(model, output, node->output_name);
}

void alloc_softmaxmerge(Model* model, const ParameterInfo* input[], ParameterInfo* output, const Node* node, CurNodeFlags* node_flags, const NodeFlags*) {
    output->slot = get_next_slot(model, input[0]);
}

void handle_softmaxmerge(Model* model, const ParameterInfo* input[], ParameterInfo* output, const Node* node, CurNodeFlags* node_flags, const NodeFlags*) {
    int axis = node_flags->softmax.axis;

    const ParameterInfo* X = input[0];
    const uint16_t softmax_length = X->dims[axis];

    uint32_t softmax_vector_idx = 0;
    uint32_t softmax_num_vectors = output->params_len / sizeof(int16_t) / softmax_length;
    uint32_t data_offset = 0;
#if INTERMITTENT
    start_cpu_counter(offsetof(Counters, progress_seeking));
    uint32_t first_unfinished_job_idx = run_recovery(model, output);
    data_offset = batch_start(job_index_to_offset(output, first_unfinished_job_idx));
    stop_cpu_counter();

    MY_ASSERT(data_offset % softmax_length == 0);
    softmax_vector_idx = data_offset / softmax_length;
#endif

    for (; softmax_vector_idx < softmax_num_vectors; softmax_vector_idx++, data_offset += softmax_length) {
        my_memcpy_from_param(model, lea_buffer, X, data_offset, softmax_length * sizeof(int16_t));

        int32_t softmax_sum = 0; // the denominator in softmax equation
        for (uint16_t softmax_idx = 0; softmax_idx < softmax_length; softmax_idx++) {
            softmax_sum += lea_buffer[softmax_idx];
        }

        // softmax_sum * (2 ** -15) = 1 / (softmax_sum_reciprocal * (2 ** -15))
        int32_t softmax_sum_reciprocal = (1LL << 30) / softmax_sum;

        // normalize the row
        my_scale_q15(lea_buffer, softmax_sum_reciprocal, /*shift=*/0, lea_buffer, softmax_length);

        my_memcpy_to_param(output, data_offset, lea_buffer, softmax_length * sizeof(int16_t), /*timer_delay=*/0, /*is_linear=*/false);
#if HAWAII
        write_hawaii_layer_footprint(model->layer_idx, softmax_length);
#endif
    }

    dump_params_debug(model, output, node->output_name);
}
