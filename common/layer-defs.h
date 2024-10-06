#pragma once

#include "config.h"

#if MY_DEBUG >= MY_DEBUG_LAYERS
typedef struct NodeFlags CurNodeFlags;
#else
typedef const struct NodeFlags CurNodeFlags;
#endif
