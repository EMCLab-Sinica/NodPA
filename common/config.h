#pragma once

// Runtime Reconfiguration
#define Fixed 0
#define DynBal 1
#define Exhaustive 2
#define RuntimeConfiguration Fixed

// Debugging
#define MY_DEBUG_NO_ASSERT 0
#define MY_DEBUG_NORMAL 1
#define MY_DEBUG_LAYERS 2
#define MY_DEBUG_VERBOSE 3

#ifndef MY_DEBUG
#define MY_DEBUG MY_DEBUG_NO_ASSERT
#endif

// Dynamic DNN
#define DYNAMIC_DNN_COARSE_GRAINED 1
#define DYNAMIC_DNN_FINE_GRAINED 2
#define DYNAMIC_DNN_ONE_INDICATOR 3
#define DYNAMIC_DNN_TWO_INDICATOR 4
#define DYNAMIC_DNN_APPROACH DYNAMIC_DNN_TWO_INDICATOR
