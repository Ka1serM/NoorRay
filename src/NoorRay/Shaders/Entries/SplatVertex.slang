import NoorRHI;
import Raster.Records;
import Raster.Splats;

[shader("vertex")]
SplatVaryings splatVertex(uint corner : SV_VulkanVertexID, uint instance : SV_InstanceID) {
    return projectSplatVertex(rootArguments<SplatArguments>(), corner, instance);
}
