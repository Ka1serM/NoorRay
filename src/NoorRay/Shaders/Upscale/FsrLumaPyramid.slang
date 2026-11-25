#include "Upscale/FsrPass.slang"
#include "fsr3upscaler/ffx_fsr3upscaler_luma_pyramid.h"

[shader("compute")]
[numthreads(256, 1, 1)]
void fsrLumaPyramid(uint3 group : SV_GroupID, uint index : SV_GroupIndex) { ComputeAutoExposure(group, index); }
