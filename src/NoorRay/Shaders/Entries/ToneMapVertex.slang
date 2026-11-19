import NoorRHI;
import Display.ToneMap;

[shader("vertex")]
ToneMapVaryings toneMapVertex(uint vertex : SV_VulkanVertexID) { return coverDisplay(vertex); }
