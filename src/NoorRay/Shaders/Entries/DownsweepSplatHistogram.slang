import NoorRHI;
import Raster.Records;
import Raster.Splats;

[shader("compute")]
[numthreads(splatGroupSize, 1, 1)]
void downsweepSplatHistogram(uint3 group : SV_GroupID, uint thread : SV_GroupIndex) { downsweepSplatTile(rootArguments<SplatArguments>(), group.x, thread); }
