import NoorRHI;
import Denoise.Denoise;

[shader("compute")]
[numthreads(8, 8, 1)]
void denoiseGather(uint3 id : SV_DispatchThreadID) { gatherDenoise(rootArguments<DenoiseArguments>(), id.xy); }
