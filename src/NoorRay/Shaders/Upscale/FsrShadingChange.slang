#include "Upscale/FsrPass.slang"
#include "fsr3upscaler/ffx_fsr3upscaler_shading_change.h"

[shader("compute")]
[numthreads(8, 8, 1)]
void fsrShadingChange(int2 position : SV_DispatchThreadID) { ShadingChange(position); }
