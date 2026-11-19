import NoorRHI;
import Raster.Records;
import Raster.Splats;

[shader("compute")]
[numthreads(splatGroupSize, 1, 1)]
void scanSplatPartials(uint thread : SV_GroupIndex) { scanSplatTiles(rootArguments<SplatArguments>(), thread); }
