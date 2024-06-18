#pragma once

#include "config.h"

#if MY_DEBUG >= MY_DEBUG_LAYERS || RuntimeConfiguration != Fixed
typedef struct NodeFlags CurNodeFlags;
#else
typedef const struct NodeFlags CurNodeFlags;
#endif
