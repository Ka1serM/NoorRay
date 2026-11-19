import NoorRHI;
import Raster.Records;
import Raster.Splats;

[shader("fragment")]
float4 splatFragment(SplatVaryings input) : SV_Target0 { return shadeSplat(input); }
