#pragma once

#define ENABLE_DEMO_COUNTERS 0

// Debugging
#define MY_DEBUG_NO_ASSERT 0
#define MY_DEBUG_NORMAL 1
#define MY_DEBUG_LAYERS 2
#define MY_DEBUG_VERBOSE 3

#ifndef MY_DEBUG
#define MY_DEBUG MY_DEBUG_NO_ASSERT
#endif
