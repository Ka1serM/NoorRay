#define FSR3UPSCALER_BIND_SRV_FRAME_INFO 3
#include "Upscale/FsrPass.slang"
#include "fsr3upscaler/ffx_fsr3upscaler_luma_instability.h"

[shader("compute")]
[numthreads(8, 8, 1)]
void fsrLumaInstability(int2 position : SV_DispatchThreadID) { LumaInstability(position); }
