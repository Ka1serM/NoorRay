import NoorRHI;
import Raster.Records;
import Raster.VisibilityBuffer;

[shader("vertex")]
VisibilityVaryings visibilityVertex(uint vertex : SV_VulkanVertexID) { return rasterizeVertex(rootArguments<VisibilityArguments>(), vertex); }
