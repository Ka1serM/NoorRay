import NoorRHI;
import Denoise.Denoise;

[shader("compute")]
[numthreads(8, 8, 1)]
void denoisePrepare(uint3 id : SV_DispatchThreadID) { prepareDenoise(rootArguments<DenoiseArguments>(), id.xy); }
