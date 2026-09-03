#pragma once

// Shared library export visibility for conv_kernels (.dll / .so).
#ifdef _WIN32
#define ML_ENGINE_EXPORT __declspec(dllexport)
#else
#define ML_ENGINE_EXPORT __attribute__((visibility("default")))
#endif

#define ML_IM2COL_EXPORT ML_ENGINE_EXPORT
