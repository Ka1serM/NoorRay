import NoorRHI;
import Raster.Records;
import Raster.Splats;

[shader("compute")]
[numthreads(splatGroupSize, 1, 1)]
void reduceSplatHistogram(uint3 group : SV_GroupID, uint thread : SV_GroupIndex) { reduceSplatTile(rootArguments<SplatArguments>(), group.x, thread); }
