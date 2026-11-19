import NoorRHI;
import Raster.Records;
import Raster.Splats;

[shader("compute")]
[numthreads(splatGroupSize, 1, 1)]
void projectSplats(uint3 group : SV_GroupID, uint thread : SV_GroupIndex) {
    const SplatArguments arguments = rootArguments<SplatArguments>();
    const uint index = getSplatBlock(group) * splatGroupSize + thread;
    if (index < arguments.count)
        projectSplat(arguments, index);
}
