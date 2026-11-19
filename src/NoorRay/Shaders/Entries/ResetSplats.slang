import NoorRHI;
import Raster.Records;
import Raster.Splats;

[shader("compute")]
[numthreads(1, 1, 1)]
void resetSplats() { resetSplatCounters(rootArguments<SplatArguments>()); }
