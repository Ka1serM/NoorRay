import NoorRHI;
import Raster.Records;
import Raster.Splats;

[shader("compute")]
[numthreads(splatGroupSize, 1, 1)]
void scatterSplatDigits(uint3 group : SV_GroupID, uint thread : SV_GroupIndex) { scatterSplatBlock(rootArguments<SplatArguments>(), getSplatBlock(group), thread); }
