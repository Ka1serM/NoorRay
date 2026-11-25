//! The options of NoorRay's FSR 3.1 passes: HDR linear input, render-resolution motion vectors, inverted depth and the reference Lanczos function.
import NoorRHI;

#define FFX_GPU 1
#define FFX_HLSL 1
#define FFX_HALF 0
#define FFX_HLSL_SM 60
#define FFX_FSR3UPSCALER_OPTION_HDR_COLOR_INPUT 1
#define FFX_FSR3UPSCALER_OPTION_LOW_RESOLUTION_MOTION_VECTORS 1
#define FFX_FSR3UPSCALER_OPTION_JITTERED_MOTION_VECTORS 0
#define FFX_FSR3UPSCALER_OPTION_INVERTED_DEPTH 1
#define FFX_FSR3UPSCALER_OPTION_REPROJECT_USE_LANCZOS_TYPE 0
#ifndef FFX_FSR3UPSCALER_OPTION_APPLY_SHARPENING
#define FFX_FSR3UPSCALER_OPTION_APPLY_SHARPENING 0
#endif
//! WGSL has no quad operations, so SPD reduces through group shared memory.
#ifdef NOORRHI_WGSL
#define FFX_SPD_NO_WAVE_OPERATIONS
#endif

#include "Upscale/FsrCallbacks.slang"
#include "fsr3upscaler/ffx_fsr3upscaler_common.h"
