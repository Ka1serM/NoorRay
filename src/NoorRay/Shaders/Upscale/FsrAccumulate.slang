#include "Upscale/FsrPass.slang"
#include "fsr3upscaler/ffx_fsr3upscaler_sample.h"
#include "fsr3upscaler/ffx_fsr3upscaler_upsample.h"
#include "fsr3upscaler/ffx_fsr3upscaler_reproject.h"
#include "fsr3upscaler/ffx_fsr3upscaler_accumulate.h"

#ifndef FSR_ACCUMULATE_ENTRY
#define FSR_ACCUMULATE_ENTRY fsrAccumulate
#endif

[shader("compute")]
[numthreads(8, 8, 1)]
void FSR_ACCUMULATE_ENTRY(int2 position : SV_DispatchThreadID) { Accumulate(position); }
