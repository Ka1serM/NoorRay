#include "Upscale/FsrPass.slang"
#include "fsr3upscaler/ffx_fsr3upscaler_prepare_inputs.h"

[shader("compute")]
[numthreads(8, 8, 1)]
void fsrPrepareInputs(int2 position : SV_DispatchThreadID) { PrepareInputs(position); }
