import NoorRHI;
import Display.ToneMap;

[shader("fragment")]
float4 toneMapFragment(ToneMapVaryings input) : SV_Target0 { return displayFilm(rootArguments<DisplayArguments>(), uint2(input.position.xy)); }
