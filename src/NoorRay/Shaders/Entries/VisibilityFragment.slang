import NoorRHI;
import Raster.Records;
import Raster.VisibilityBuffer;

[shader("fragment")]
uint visibilityFragment(VisibilityVaryings input) : SV_Target0 { return input.id; }
