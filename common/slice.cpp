#include "cnn_common.h"
#include "my_debug.h"

void alloc_slice(struct Model *model, const struct ParameterInfo *input[], struct ParameterInfo *output, const struct Node* node, CurNodeFlags*, const NodeFlags*) {
    const ParameterInfo *X = input[0], *start = input[1], *end = input[2], *axes = input[3];

    uint16_t input_start = get_int64_param(start, 0);
    uint16_t input_end = get_int64_param(end, 0);
    uint16_t input_axes = get_int64_param(axes, 0);

    output->slot = get_next_slot(model, X);

    output->params_len = sizeof(int16_t);

    for(uint8_t dim_idx = 0; dim_idx < 3; dim_idx++) {
        if(dim_idx == input_axes){
            output->dims[dim_idx] = input_end-input_start;
        }
        else{
            output->dims[dim_idx] = X->dims[dim_idx];
        }
        output->params_len *= output->dims[dim_idx];
    }
}

void handle_slice(struct Model *model, const struct ParameterInfo *input[], struct ParameterInfo *output, const struct Node* node, CurNodeFlags*, const NodeFlags*) {
    my_printf_debug("Slice!" NEWLINE);

    const ParameterInfo *X = input[0], *start = input[1], *end = input[2];
    uint16_t input_start = get_int64_param(start, 0);
    uint16_t input_end = get_int64_param(end, 0);

    uint32_t output_offset = 0;

    for (uint16_t idx0 = input_start; idx0 < input_end; idx0++) {
        for (uint16_t idx1 = 0; idx1 < X->dims[1]; idx1++) {
            for (uint16_t idx2 = 0; idx2 < X->dims[2]; idx2++) {
                uint32_t input_offset = idx0 * X->dims[1] * X->dims[2] + idx1 * X->dims[2] + idx2;
                int16_t input_val = get_q15_param(model, X, input_offset);
                put_q15_param(output, output_offset, input_val, /*is_linear=*/false);
                output_offset++;
            }
        }
     }
}
