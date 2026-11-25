import NoorRHI;
import Denoise.Denoise;

[shader("compute")]
[numthreads(8, 8, 1)]
void denoiseClear(uint3 id : SV_DispatchThreadID) { clearDenoise(rootArguments<DenoiseClearArguments>(), id.xy); }
