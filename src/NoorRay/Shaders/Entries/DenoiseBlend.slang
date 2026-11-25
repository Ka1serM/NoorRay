import NoorRHI;
import Denoise.Denoise;

[shader("compute")]
[numthreads(8, 8, 1)]
void denoiseBlend(uint3 id : SV_DispatchThreadID) { blendDenoise(rootArguments<DenoiseArguments>(), id.xy); }
