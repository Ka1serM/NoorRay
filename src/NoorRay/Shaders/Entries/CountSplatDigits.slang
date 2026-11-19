import NoorRHI;
import Raster.Records;
import Raster.Splats;

[shader("compute")]
[numthreads(splatGroupSize, 1, 1)]
void countSplatDigits(uint3 group : SV_GroupID, uint thread : SV_GroupIndex) { countSplatBlock(rootArguments<SplatArguments>(), getSplatBlock(group), thread); }
