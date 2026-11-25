import NoorRHI;
import Denoise.Denoise;

[shader("compute")]
[numthreads(8, 8, 1)]
void denoiseComposite(uint3 id : SV_DispatchThreadID) { compositeDenoise(rootArguments<DenoiseArguments>(), id.xy); }
