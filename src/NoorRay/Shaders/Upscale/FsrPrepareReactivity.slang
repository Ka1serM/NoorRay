#include "Upscale/FsrPass.slang"
#include "fsr3upscaler/ffx_fsr3upscaler_prepare_reactivity.h"

[shader("compute")]
[numthreads(8, 8, 1)]
void fsrPrepareReactivity(int2 position : SV_DispatchThreadID) { PrepareReactivity(position); }
