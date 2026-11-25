#include "Upscale/FsrPass.slang"
#include "fsr3upscaler/ffx_fsr3upscaler_rcas.h"

[shader("compute")]
[numthreads(64, 1, 1)]
void fsrRcas(uint3 thread : SV_GroupThreadID, uint3 group : SV_GroupID, uint3 position : SV_DispatchThreadID) { RCAS(thread, group, position); }
