#include "Redistancing.slang"
#include "SurfaceTracing.slang"

[shader("compute")]
[numthreads(8, 8, 1)]
void render(uint3 id : SV_DispatchThreadID) {
    const Arguments arguments = rootArguments<Arguments>();
    if (id.x >= arguments.width || id.y >= arguments.height)
        return;
    renderPixel(arguments, id.xy);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void backward(uint3 id : SV_DispatchThreadID) {
    const Arguments arguments = rootArguments<Arguments>();
    if (id.x >= arguments.width || id.y >= arguments.height)
        return;
    backwardPixel(arguments, id.xy);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void seed(uint3 id : SV_DispatchThreadID) {
    const Arguments arguments = rootArguments<Arguments>();
    for (uint index = id.x; index < arguments.grid.size.w; index += arguments.threads)
        seedNode(arguments, index);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void flood(uint3 id : SV_DispatchThreadID) {
    const Arguments arguments = rootArguments<Arguments>();
    for (uint index = id.x; index < arguments.grid.size.w; index += arguments.threads)
        floodNode(arguments, index);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void redistance(uint3 id : SV_DispatchThreadID) {
    const Arguments arguments = rootArguments<Arguments>();
    for (uint index = id.x; index < arguments.grid.size.w; index += arguments.threads)
        applyDistance(arguments, index);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void smooth(uint3 id : SV_DispatchThreadID) {
    const Arguments arguments = rootArguments<Arguments>();
    for (uint index = id.x; index < arguments.grid.size.w; index += arguments.threads)
        addCurvatureGradient(arguments, index);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void clear(uint3 id : SV_DispatchThreadID) {
    const Arguments arguments = rootArguments<Arguments>();
    for (uint index = id.x; index < arguments.count; index += arguments.threads)
        arguments.clearTarget[index] = 0.0;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void resample(uint3 id : SV_DispatchThreadID) {
    const Arguments arguments = rootArguments<Arguments>();
    for (uint index = id.x; index < arguments.grid.size.w; index += arguments.threads)
        resampleNode(arguments, index);
}
