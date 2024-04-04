#include <cstdint>
#include "cnn_common.h"
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

    uint16_t buffer_size = 256;
    int16_t *buffer_x = lea_buffer,
            *buffer_y = buffer_x + buffer_size;
    my_memcpy_from_param(model, buffer_y, Y, 0, sizeof(int16_t));

    int16_t scaleFract;
    uint8_t shift;
    float_to_scale_params(&scaleFract, &shift, q15_to_float(*buffer_y, ValueInfo(Y)));

    for (; data_offset < output->params_len / sizeof(int16_t); data_offset += buffer_size) {
        uint16_t cur_buffer_size = MIN_VAL(output->params_len / sizeof(int16_t) - data_offset, buffer_size);
        my_printf_debug("data_offset=%d" NEWLINE, data_offset);
        my_memcpy_from_param(model, buffer_x, X, data_offset, cur_buffer_size * sizeof(int16_t));

        my_scale_q15(buffer_x, scaleFract, shift, buffer_x, buffer_size);
        my_printf_debug("After mul" NEWLINE);
        dump_matrix_debug(buffer_x, cur_buffer_size, ValueInfo(output), false);

        my_memcpy_to_param(output, data_offset, buffer_x, cur_buffer_size * sizeof(int16_t), 0, true);
    }

    dump_params_debug(model, output, node->output_name);
}
