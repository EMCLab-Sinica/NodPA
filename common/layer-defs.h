#pragma once

#include "dynbal-config.h"

#if RuntimeConfiguration != Fixed
typedef struct NodeFlags CurNodeFlags;
#else
typedef const struct NodeFlags CurNodeFlags;
#endif
