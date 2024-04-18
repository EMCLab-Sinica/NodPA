#include "cnn_common.h"
#include "layer-defs.h"
#include "my_debug.h"

#include <cstdint>

void handle_dropout(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node* node, CurNodeFlags* node_flags, const NodeFlags* orig_node_flags) {
    const ParameterInfo* ratio = input[1];

    int16_t ratio_val = get_q15_param(model, ratio, /*offset_in_word=*/0);

    MY_ASSERT(ratio_val == 0, "Only no-op dropout is implemented.");
}
