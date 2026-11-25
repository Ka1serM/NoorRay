import NoorRHI;
import Display.FilmAlpha;
import Display.Records;

[shader("compute")]
[numthreads(8, 8, 1)]
void readImage(uint3 id : SV_DispatchThreadID) {
    const ReadImageArguments arguments = rootArguments<ReadImageArguments>();
    if (id.x >= arguments.width || id.y >= arguments.height)
        return;
    const float alpha = filmAlpha(arguments.film, uint2(arguments.filmWidth, arguments.filmHeight), id.xy, uint2(arguments.width, arguments.height));
    arguments.destination[id.y * arguments.width + id.x] = float4(arguments.image.Load(int2(id.xy)).xyz, alpha);
}
