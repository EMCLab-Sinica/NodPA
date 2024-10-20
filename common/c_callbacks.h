#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif
struct ParameterInfo;
typedef void (*data_preservation_func)(struct ParameterInfo *param, uint32_t offset_in_word, const void *src, size_t n, uint16_t timer_delay, bool is_linear);
#ifdef __cplusplus
}
#endif
