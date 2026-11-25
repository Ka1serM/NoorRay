#include "Rasterizing.h"

typealias Arguments = Rasterizing::Arguments;

struct Texture {
    Ptr<float> texels;
    int2 size;
};

float3 fetch(Texture texture, int2 texel) {
    const uint index = (texel.y * texture.size.x + texel.x) * 3;
    return float3(texture.texels[index], texture.texels[index + 1], texture.texels[index + 2]);
}

[Differentiable]
float3 sampleBilinear(no_diff Texture texture, float2 uv) {
    const float2 texel = (uv - no_diff floor(uv)) * float2(texture.size) - 0.5;
    const float2 base = no_diff floor(texel);
    const float2 weight = texel - base;
    int2 low = int2(base);
    int2 high = low + 1;
    low += select(low < 0, texture.size, int2(0));
    high -= select(high >= texture.size, texture.size, int2(0));
    const float3 bottom = lerp(no_diff fetch(texture, low), no_diff fetch(texture, int2(high.x, low.y)), weight.x);
    const float3 top = lerp(no_diff fetch(texture, int2(low.x, high.y)), no_diff fetch(texture, high), weight.x);
    return lerp(bottom, top, weight.y);
}

float3 getEdgeFunctions(float4 p0, float4 p1, float4 p2, float2 pixel) {
    const float2 q0 = p0.xy - pixel * p0.w;
    const float2 q1 = p1.xy - pixel * p1.w;
    const float2 q2 = p2.xy - pixel * p2.w;
    return float3(q1.x * q2.y - q1.y * q2.x, q2.x * q0.y - q2.y * q0.x, q0.x * q1.y - q0.y * q1.x);
}

[Differentiable]
float2 getBarycentrics(float4 p0, float4 p1, float4 p2, no_diff float2 pixel) {
    const float2 q0 = p0.xy - pixel * p0.w;
    const float2 q1 = p1.xy - pixel * p1.w;
    const float2 q2 = p2.xy - pixel * p2.w;
    const float a0 = q1.x * q2.y - q1.y * q2.x;
    const float a1 = q2.x * q0.y - q2.y * q0.x;
    const float a2 = q0.x * q1.y - q0.y * q1.x;
    return float2(a0, a1) / (a0 + a1 + a2);
}

[Differentiable]
float3 shade(float2 barycentrics, no_diff float2 uv0, no_diff float2 uv1, no_diff float2 uv2, no_diff Texture texture) {
    const float2 uv = barycentrics.x * uv0 + barycentrics.y * uv1 + (1.0 - barycentrics.x - barycentrics.y) * uv2;
    return sampleBilinear(texture, uv);
}

[Differentiable]
float getEdgeCrossing(float4 p1, float4 p2, no_diff float2 center, no_diff float2 halfSize, no_diff bool down) {
    float2 a = p1.xy / p1.w;
    float2 b = p2.xy / p2.w;
    float2 scale = halfSize;
    float2 offset = center;
    if (down) {
        a = a.yx;
        b = b.yx;
        scale = scale.yx;
        offset = offset.yx;
    }
    const float2 v1 = a * scale - offset;
    const float2 v2 = b * scale - offset;
    const float2 delta = v2 - v1;
    return (v1.x * delta.y - v1.y * delta.x) / delta.y;
}

uint3 getTriangle(Arguments arguments, uint triangle) {
    return uint3(arguments.triangles[triangle * 3], arguments.triangles[triangle * 3 + 1], arguments.triangles[triangle * 3 + 2]);
}

Texture getTexture(Arguments arguments) {
    return {arguments.texture, int2(arguments.textureWidth, arguments.textureHeight)};
}

float2 getPixelCenter(Arguments arguments, uint2 pixel) {
    return (float2(pixel) * 2.0 + 1.0) / float2(arguments.width, arguments.height) - 1.0;
}

float3 loadColor(Arguments arguments, uint pixel) {
    return float3(arguments.color[pixel * 3], arguments.color[pixel * 3 + 1], arguments.color[pixel * 3 + 2]);
}

float3 loadImageGradient(Arguments arguments, uint pixel) {
    return float3(arguments.imageGradient[pixel * 3], arguments.imageGradient[pixel * 3 + 1], arguments.imageGradient[pixel * 3 + 2]);
}

void addColor(AtomicPtr<float> image, uint pixel, float3 value) {
    atomicAdd(image + pixel * 3, value.x);
    atomicAdd(image + pixel * 3 + 1, value.y);
    atomicAdd(image + pixel * 3 + 2, value.z);
}

void addPosition(Arguments arguments, uint vertex, float4 value) {
    const AtomicPtr<float> target = arguments.positionGradient + vertex * 4;
    atomicAdd(target, value.x);
    atomicAdd(target + 1, value.y);
    atomicAdd(target + 3, value.w);
}

bool isSameSign(float a, float b) { return (asint(a) ^ asint(b)) >= 0; }

bool isRationalGreater(float n0, float n1, float d0, float d1) { return (n0 * d1 > n1 * d0) == isSameSign(d0, d1); }

uint getMaximumIndex(float3 numerators, float3 denominators) {
    const bool greater10 = isRationalGreater(numerators.y, numerators.x, denominators.y, denominators.x);
    const bool greater20 = isRationalGreater(numerators.z, numerators.x, denominators.z, denominators.x);
    const bool greater21 = isRationalGreater(numerators.z, numerators.y, denominators.z, denominators.y);
    if (greater20 && greater21)
        return 2;
    return greater10 ? 1 : 0;
}

struct Edge {
    bool found;
    float alpha;
    uint triangle;
    uint index;
    uint pixel0;
    uint pixel1;
    float2 center;
};

Edge findEdge(Arguments arguments, uint2 pixel, bool down) {
    Edge edge = {};
    edge.pixel0 = pixel.y * arguments.width + pixel.x;
    edge.pixel1 = edge.pixel0 + (down ? arguments.width : 1);
    const float4 raster0 = arguments.raster[edge.pixel0];
    const float4 raster1 = arguments.raster[edge.pixel1];
    if (raster0.w == raster1.w)
        return edge;
    const int triangle0 = int(raster0.w) - 1;
    const int triangle1 = int(raster1.w) - 1;
    int triangle = triangle0 >= 0 ? triangle0 : triangle1;
    if (triangle0 >= 0 && triangle1 >= 0)
        triangle = raster0.z < raster1.z ? triangle0 : triangle1;
    const bool first = triangle == triangle0;
    const uint2 owner = first ? pixel : pixel + (down ? uint2(0, 1) : uint2(1, 0));
    const uint3 vertices = getTriangle(arguments, triangle);
    const int3 opposites = int3(arguments.opposites[triangle * 3], arguments.opposites[triangle * 3 + 1], arguments.opposites[triangle * 3 + 2]);
    const float2 halfSize = float2(arguments.width, arguments.height) * 0.5;
    const float2 center = float2(owner) + 0.5 - halfSize;
    float2 v[3];
    float2 o[3];
    [ForceUnroll]
    for (uint index = 0; index < 3; ++index) {
        const float4 position = arguments.positions[vertices[index]];
        const float4 opposite = opposites[index] < 0 ? position : arguments.positions[opposites[index]];
        v[index] = position.xy / position.w * halfSize - center;
        o[index] = opposite.xy / opposite.w * halfSize - center;
    }
    const float area = (v[1].x - v[0].x) * (v[2].y - v[0].y) - (v[2].x - v[0].x) * (v[1].y - v[0].y);
    const float3 wings = float3((v[1].x - o[0].x) * (v[2].y - o[0].y) - (v[2].x - o[0].x) * (v[1].y - o[0].y),
                                (v[2].x - o[1].x) * (v[0].y - o[1].y) - (v[0].x - o[1].x) * (v[2].y - o[1].y),
                                (v[0].x - o[2].x) * (v[1].y - o[2].y) - (v[1].x - o[2].x) * (v[0].y - o[2].y));
    const bool3 silhouette = bool3(isSameSign(wings.x, area), isSameSign(wings.y, area), isSameSign(wings.z, area));
    if (!any(silhouette))
        return edge;
    if (down) {
        [ForceUnroll]
        for (uint index = 0; index < 3; ++index)
            v[index] = v[index].yx;
    }
    float3 dx = float3(v[2].x - v[1].x, v[0].x - v[2].x, v[1].x - v[0].x);
    float3 dy = float3(v[2].y - v[1].y, v[0].y - v[2].y, v[1].y - v[0].y);
    const float side = first ? 1.0 : -1.0;
    float3 distances = side * float3(v[1].x * dy.x - v[1].y * dx.x, v[2].x * dy.y - v[2].y * dx.y, v[0].x * dy.z - v[0].y * dx.z);
    [ForceUnroll]
    for (uint index = 0; index < 3; ++index) {
        if (!isSameSign(v[(index + 1) % 3].y, v[(index + 2) % 3].y))
            continue;
        distances[index] = -3.402823466e+38;
        dy[index] = 1.0;
    }
    const uint index = getMaximumIndex(distances, dy);
    if (!silhouette[index] || abs(dy[index]) < abs(dx[index]))
        return edge;
    const float crossing = distances[index] / dy[index];
    const float tolerance = 0.0625;
    if (crossing <= -tolerance || crossing >= 1.0 + tolerance)
        return edge;
    edge.found = true;
    edge.alpha = side * (0.5 - clamp(crossing, 0.0, 1.0));
    edge.triangle = triangle;
    edge.index = index;
    edge.center = center;
    return edge;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void clear(uint3 id : SV_DispatchThreadID) {
    const Arguments arguments = rootArguments<Arguments>();
    const uint pixels = arguments.width * arguments.height * 3;
    if (id.x < pixels) {
        atomicStore(arguments.image + id.x, 0.0);
        atomicStore(arguments.colorGradient + id.x, 0.0);
    }
    if (id.x < arguments.vertexCount * 4)
        atomicStore(arguments.positionGradient + id.x, 0.0);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void rasterize(uint3 id : SV_DispatchThreadID) {
    const Arguments arguments = rootArguments<Arguments>();
    if (any(id.xy >= uint2(arguments.width, arguments.height)))
        return;
    const float2 center = getPixelCenter(arguments, id.xy);
    float4 result = float4(0.0);
    float depth = 2.0;
    for (uint triangle = 0; triangle < arguments.triangleCount; ++triangle) {
        const uint3 vertices = getTriangle(arguments, triangle);
        const float4 p0 = arguments.positions[vertices.x];
        const float4 p1 = arguments.positions[vertices.y];
        const float4 p2 = arguments.positions[vertices.z];
        const float3 edges = getEdgeFunctions(p0, p1, p2, center);
        const float area = edges.x + edges.y + edges.z;
        if (area == 0.0 || any(edges * area < 0.0))
            continue;
        const float z = dot(float3(p0.z, p1.z, p2.z), edges) / dot(float3(p0.w, p1.w, p2.w), edges);
        if (z < -1.0 || z > 1.0 || z >= depth)
            continue;
        depth = z;
        float2 barycentrics = saturate(edges.xy / area);
        barycentrics /= max(barycentrics.x + barycentrics.y, 1.0);
        result = float4(barycentrics, z, float(triangle + 1));
    }
    arguments.raster[id.y * arguments.width + id.x] = result;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void shadePixels(uint3 id : SV_DispatchThreadID) {
    const Arguments arguments = rootArguments<Arguments>();
    if (any(id.xy >= uint2(arguments.width, arguments.height)))
        return;
    const uint pixel = id.y * arguments.width + id.x;
    const float4 raster = arguments.raster[pixel];
    float3 color = arguments.background.xyz;
    if (raster.w > 0.0) {
        const uint triangle = uint(raster.w) - 1;
        const uint3 uvIndices = uint3(arguments.uvTriangles[triangle * 3], arguments.uvTriangles[triangle * 3 + 1], arguments.uvTriangles[triangle * 3 + 2]);
        color = shade(raster.xy, arguments.uvs[uvIndices.x], arguments.uvs[uvIndices.y], arguments.uvs[uvIndices.z], getTexture(arguments));
    }
    arguments.color[pixel * 3] = color.x;
    arguments.color[pixel * 3 + 1] = color.y;
    arguments.color[pixel * 3 + 2] = color.z;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void antialias(uint3 id : SV_DispatchThreadID) {
    const Arguments arguments = rootArguments<Arguments>();
    if (any(id.xy >= uint2(arguments.width, arguments.height)))
        return;
    const uint pixel = id.y * arguments.width + id.x;
    addColor(arguments.image, pixel, loadColor(arguments, pixel));
    if (arguments.antialiasing == 0)
        return;
    [ForceUnroll]
    for (uint direction = 0; direction < 2; ++direction) {
        const bool down = direction == 1;
        if ((down ? id.y + 1 : id.x + 1) >= (down ? arguments.height : arguments.width))
            continue;
        const Edge edge = findEdge(arguments, id.xy, down);
        if (!edge.found)
            continue;
        const float3 difference = loadColor(arguments, edge.pixel1) - loadColor(arguments, edge.pixel0);
        addColor(arguments.image, edge.alpha > 0.0 ? edge.pixel0 : edge.pixel1, edge.alpha * difference);
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void antialiasBackward(uint3 id : SV_DispatchThreadID) {
    const Arguments arguments = rootArguments<Arguments>();
    if (any(id.xy >= uint2(arguments.width, arguments.height)))
        return;
    const uint pixel = id.y * arguments.width + id.x;
    addColor(arguments.colorGradient, pixel, loadImageGradient(arguments, pixel));
    if (arguments.antialiasing == 0)
        return;
    const float2 halfSize = float2(arguments.width, arguments.height) * 0.5;
    [ForceUnroll]
    for (uint direction = 0; direction < 2; ++direction) {
        const bool down = direction == 1;
        if ((down ? id.y + 1 : id.x + 1) >= (down ? arguments.height : arguments.width))
            continue;
        const Edge edge = findEdge(arguments, id.xy, down);
        if (!edge.found)
            continue;
        const float3 gradient = loadImageGradient(arguments, edge.alpha > 0.0 ? edge.pixel0 : edge.pixel1);
        addColor(arguments.colorGradient, edge.pixel0, -edge.alpha * gradient);
        addColor(arguments.colorGradient, edge.pixel1, edge.alpha * gradient);
        const float weight = dot(gradient, loadColor(arguments, edge.pixel1) - loadColor(arguments, edge.pixel0));
        if (weight == 0.0 || abs(edge.alpha) >= 0.5)
            continue;
        const uint3 vertices = getTriangle(arguments, edge.triangle);
        const uint vertex1 = vertices[(edge.index + 1) % 3];
        const uint vertex2 = vertices[(edge.index + 2) % 3];
        var p1 = diffPair(arguments.positions[vertex1]);
        var p2 = diffPair(arguments.positions[vertex2]);
        bwd_diff(getEdgeCrossing)(p1, p2, edge.center, halfSize, down, -weight);
        addPosition(arguments, vertex1, p1.d);
        addPosition(arguments, vertex2, p2.d);
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void shadeBackward(uint3 id : SV_DispatchThreadID) {
    const Arguments arguments = rootArguments<Arguments>();
    if (any(id.xy >= uint2(arguments.width, arguments.height)))
        return;
    const uint pixel = id.y * arguments.width + id.x;
    const float4 raster = arguments.raster[pixel];
    if (raster.w <= 0.0)
        return;
    const uint triangle = uint(raster.w) - 1;
    const uint3 vertices = getTriangle(arguments, triangle);
    const uint3 uvIndices = uint3(arguments.uvTriangles[triangle * 3], arguments.uvTriangles[triangle * 3 + 1], arguments.uvTriangles[triangle * 3 + 2]);
    const float3 gradient = float3(atomicLoad(arguments.colorGradient + pixel * 3), atomicLoad(arguments.colorGradient + pixel * 3 + 1),
                                   atomicLoad(arguments.colorGradient + pixel * 3 + 2));
    var barycentrics = diffPair(raster.xy);
    bwd_diff(shade)(barycentrics, arguments.uvs[uvIndices.x], arguments.uvs[uvIndices.y], arguments.uvs[uvIndices.z], getTexture(arguments), gradient);
    var p0 = diffPair(arguments.positions[vertices.x]);
    var p1 = diffPair(arguments.positions[vertices.y]);
    var p2 = diffPair(arguments.positions[vertices.z]);
    bwd_diff(getBarycentrics)(p0, p1, p2, getPixelCenter(arguments, id.xy), barycentrics.d);
    addPosition(arguments, vertices.x, p0.d);
    addPosition(arguments, vertices.y, p1.d);
    addPosition(arguments, vertices.z, p2.d);
}
