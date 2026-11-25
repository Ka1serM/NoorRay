#include "Upscale/FsrPass.slang"
#include "fsr3upscaler/ffx_fsr3upscaler_shading_change_pyramid.h"

[shader("compute")]
[numthreads(256, 1, 1)]
void fsrShadingChangePyramid(uint3 group : SV_GroupID, uint index : SV_GroupIndex) { ComputeShadingChangePyramid(group, index); }
