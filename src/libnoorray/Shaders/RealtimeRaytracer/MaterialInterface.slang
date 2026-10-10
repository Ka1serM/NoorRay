// The contract between the realtime renderer and the materials it shades.
//
// MaterialX documents are generated into Slang modules against this module
// (Materials/MaterialX/SlangMaterialGenerator.cpp), which the runtime links
// into one hit group per distinct material (MaterialHit.slang). The material's
// closest hit returns the closures it evaluates, and the ray-generation stage
// shades them with its own GGX model.
//
// MaterialX's closure types are defined here, not by MaterialX's hardware
// libraries: a BSDF is not evaluated against lights inside the material but
// returned as the lobes the renderer samples. Every BSDF reduces to a
// Lambertian diffuse lobe plus one GGX reflection lobe. Closure operations
// combine lobes linearly; each lobe's roughness and normal are averaged by
// the weight the lobe contributes.

import noorrhi;

// Where a material is evaluated. World space; directions are normalized.
struct MaterialGeometry
{
    float3 position;
    float3 normal;
    float3 tangent;
    float3 bitangent;
    float2 uv;
    // Every UV channel of the mesh, uvCount per vertex, and the hit
    // triangle's vertices in it.
    float2* uvs;
    uint uvCount;
    uint3 uvVertices;
    float2 uvBarycentrics;
    float4 color;
    // Object-space bounds of the instance's mesh.
    float3 boundsMin;
    float3 boundsMax;
    // Screen-space derivatives, per render pixel along the image's x and y,
    // from the ray differentials (the *_ddx and *_ddy geometric properties).
    float3 positionDx;
    float3 positionDy;
    float3 normalDx;
    float3 normalDy;
    float3 tangentDx;
    float3 tangentDy;
    float3 bitangentDx;
    float3 bitangentDy;
    float2 uvDx;
    float2 uvDy;
    float2 uvBarycentricsDx;
    float2 uvBarycentricsDy;
    float4 colorDx;
    float4 colorDy;
};

// The channel's three vertex UVs, or none where the mesh lacks the channel.
bool materialUvVertices(MaterialGeometry geometry, uint channel, out float2 a, out float2 b,
    out float2 c)
{
    a = b = c = float2(0.0);
    if (channel >= geometry.uvCount)
        return false;
    a = geometry.uvs[geometry.uvVertices.x * geometry.uvCount + channel];
    b = geometry.uvs[geometry.uvVertices.y * geometry.uvCount + channel];
    c = geometry.uvs[geometry.uvVertices.z * geometry.uvCount + channel];
    return true;
}

// Fetch only the channel a generated MaterialX texcoord node actually uses.
float2 materialUv(MaterialGeometry geometry, uint channel)
{
    if (channel == 0u) return geometry.uv;
    float2 a, b, c;
    if (!materialUvVertices(geometry, channel, a, b, c))
        return float2(0.0);
    return a * (1.0 - geometry.uvBarycentrics.x - geometry.uvBarycentrics.y)
        + b * geometry.uvBarycentrics.x + c * geometry.uvBarycentrics.y;
}

// The channel's screen-space derivative along the image's x (axis 0) or y.
float2 materialUvDerivative(MaterialGeometry geometry, uint channel, uint axis)
{
    if (channel == 0u) return axis == 0u ? geometry.uvDx : geometry.uvDy;
    float2 a, b, c;
    if (!materialUvVertices(geometry, channel, a, b, c))
        return float2(0.0);
    const float2 delta = axis == 0u ? geometry.uvBarycentricsDx : geometry.uvBarycentricsDy;
    return (b - a) * delta.x + (c - a) * delta.y;
}

// An instance transform in MaterialX's convention: mx_matrix_mul(m, v) is
// mul(v, m), so the rows are the images of the basis vectors and the origin.
float4x4 materialMatrix(float4x3 m)
{
    return float4x4(float4(m[0], 0.0), float4(m[1], 0.0), float4(m[2], 0.0), float4(m[3], 1.0));
}

float closureLuminance(float3 value)
{
    return dot(value, float3(0.2126, 0.7152, 0.0722));
}

struct BSDF
{
    float3 diffuse = float3(0.0);
    // Fresnel reflectance of the GGX lobe at normal and grazing incidence.
    float3 specularF0 = float3(0.0);
    float3 specularF90 = float3(0.0);
    // Weighted sums, resolved by the renderer: the GGX roughness and the
    // shading normal.
    float roughnessSum = 0.0;
    float roughnessWeight = 0.0;
    float3 normalSum = float3(0.0);

    // Directional average of the lobes' reflectance: what a layer on top
    // takes away from the layers below it.
    float3 albedo()
    {
        return diffuse + specularF0 + (specularF90 - specularF0) / 21.0;
    }

    float roughness()
    {
        return roughnessWeight > 0.0 ? roughnessSum / roughnessWeight : 1.0;
    }

    float3 normal(float3 fallback)
    {
        return dot(normalSum, normalSum) > 1.0e-12 ? normalize(normalSum) : fallback;
    }
};

// Emission is radiance leaving the surface, the same in every direction.
typealias EDF = float3;
// Volumes are not rendered; a VDF carries nothing.
struct VDF
{
};

struct surfaceshader
{
    BSDF bsdf = BSDF();
    float3 emission = float3(0.0);
    float opacity = 1.0;
};

// Perceptual roughness of MaterialX's anisotropic GGX alpha.
float closureRoughness(float2 alpha)
{
    return sqrt(saturate(0.5 * (alpha.x + alpha.y)));
}

// A lobe of zero weight is empty rather than weighted by zero: MaterialX
// graphs feed disabled lobes inputs that may not be finite (standard_surface
// derives a conductor from the base color even at metalness 0), and zero
// times NaN would poison the whole closure.
BSDF diffuseClosure(float3 albedo, float3 normal)
{
    BSDF result = BSDF();
    if (!any(albedo > 0.0))
        return result;
    result.diffuse = albedo;
    result.normalSum = normal * closureLuminance(albedo);
    return result;
}

BSDF reflectionClosure(float3 f0, float3 f90, float2 alpha, float3 normal)
{
    BSDF result = BSDF();
    if (!any(f90 > 0.0) && !any(f0 > 0.0))
        return result;
    result.specularF0 = f0;
    result.specularF90 = f90;
    const float weight = closureLuminance(result.albedo());
    result.roughnessSum = closureRoughness(alpha) * weight;
    result.roughnessWeight = weight;
    result.normalSum = normal * weight;
    return result;
}

BSDF scaleClosure(BSDF closure, float3 weight)
{
    if (!any(weight > 0.0))
        return BSDF();
    const float scalar = closureLuminance(weight);
    closure.diffuse *= weight;
    closure.specularF0 *= weight;
    closure.specularF90 *= weight;
    closure.roughnessSum *= scalar;
    closure.roughnessWeight *= scalar;
    closure.normalSum *= scalar;
    return closure;
}

BSDF addClosures(BSDF a, BSDF b)
{
    BSDF result = BSDF();
    result.diffuse = a.diffuse + b.diffuse;
    result.specularF0 = a.specularF0 + b.specularF0;
    result.specularF90 = a.specularF90 + b.specularF90;
    result.roughnessSum = a.roughnessSum + b.roughnessSum;
    result.roughnessWeight = a.roughnessWeight + b.roughnessWeight;
    result.normalSum = a.normalSum + b.normalSum;
    return result;
}

// Scatter modes of MaterialX's enumeration "R,T,RT". Transmission is not
// rendered: a transmission-only lobe reflects nothing.
static const int ClosureScatterTransmission = 1;

float dielectricF0(float ior)
{
    const float r = (ior - 1.0) / (ior + 1.0);
    return r * r;
}

// ---------------------------------------------------------------------------
// MaterialX pbrlib and stdlib closure nodes, in nodedef input order.
// ---------------------------------------------------------------------------

BSDF mx_oren_nayar_diffuse_bsdf(float weight, float3 color, float roughness,
    float3 normal, bool energyCompensation)
{
    return diffuseClosure(weight * color, normal);
}

BSDF mx_burley_diffuse_bsdf(float weight, float3 color, float roughness, float3 normal)
{
    return diffuseClosure(weight * color, normal);
}

BSDF mx_subsurface_bsdf(float weight, float3 color, float3 radius, float anisotropy,
    float3 normal)
{
    return diffuseClosure(weight * color, normal);
}

BSDF mx_dielectric_bsdf(float weight, float3 tint, float ior, float2 roughness,
    float3 normal, int scatterMode)
{
    if (!(weight > 0.0) || scatterMode == ClosureScatterTransmission)
        return BSDF();
    return reflectionClosure(weight * tint * dielectricF0(ior), weight * tint,
        roughness, normal);
}

BSDF mx_conductor_bsdf(float weight, float3 ior, float3 extinction, float2 roughness,
    float3 normal)
{
    if (!(weight > 0.0))
        return BSDF();
    const float3 k2 = extinction * extinction;
    const float3 f0 = ((ior - 1.0) * (ior - 1.0) + k2) / ((ior + 1.0) * (ior + 1.0) + k2);
    return reflectionClosure(weight * f0, float3(weight), roughness, normal);
}

BSDF mx_generalized_schlick_bsdf(float weight, float3 color0, float3 color90,
    float2 roughness, float3 normal, int scatterMode)
{
    if (!(weight > 0.0) || scatterMode == ClosureScatterTransmission)
        return BSDF();
    return reflectionClosure(weight * color0, weight * color90, roughness, normal);
}

BSDF mx_layer_bsdf(BSDF top, BSDF base)
{
    return addClosures(top, scaleClosure(base, saturate(1.0 - top.albedo())));
}

BSDF mx_mix_bsdf(BSDF fg, BSDF bg, float mix)
{
    return addClosures(scaleClosure(fg, float3(mix)), scaleClosure(bg, float3(1.0 - mix)));
}

surfaceshader mx_surface(BSDF bsdf, EDF edf, float opacity)
{
    surfaceshader result = surfaceshader();
    result.bsdf = bsdf;
    result.emission = edf;
    result.opacity = opacity;
    return result;
}

surfaceshader mx_surface_unlit(float emission, float3 emissionColor, float opacity)
{
    surfaceshader result = surfaceshader();
    result.emission = emission * emissionColor;
    result.opacity = opacity;
    return result;
}

surfaceshader mx_mix_surfaceshader(surfaceshader fg, surfaceshader bg, float mix)
{
    surfaceshader result = surfaceshader();
    result.bsdf = mx_mix_bsdf(fg.bsdf, bg.bsdf, mix);
    result.emission = lerp(bg.emission, fg.emission, mix);
    result.opacity = lerp(bg.opacity, fg.opacity, mix);
    return result;
}

// ---------------------------------------------------------------------------
// Textures. MaterialX filename parameters are bindless texture
// indices; 0 means no texture and samples white.
// ---------------------------------------------------------------------------

struct SamplerTexture2D
{
    uint handle;
}

// Set by the hit shader for this invocation. UV units covered by one render
// pixel: the longer of the ray differentials' UV derivatives.
static float materialTextureFootprint = 0.0;
static uint materialSamplerHandle = 0u;

float4 sampleMaterialTexture(SamplerTexture2D tex, float2 uv, float lod)
{
    if (tex.handle == 0u || materialSamplerHandle == 0u)
        return float4(1.0);
    Texture2D<float4> image = ResourceDescriptorHeap[NonUniformResourceIndex(tex.handle)];
    SamplerState sampler = SamplerDescriptorHeap[NonUniformResourceIndex(materialSamplerHandle)];
    return image.SampleLevel(sampler, uv, max(lod, 0.0));
}

float4 texture(SamplerTexture2D tex, float2 uv)
{
    if (tex.handle == 0u)
        return float4(1.0);
    Texture2D<float4> image = ResourceDescriptorHeap[NonUniformResourceIndex(tex.handle)];
    uint width = 0u;
    uint height = 0u;
    image.GetDimensions(width, height);
    if (width == 0u || height == 0u)
        return float4(1.0);
    const float texelsPerPixel = materialTextureFootprint * float(max(width, height));
    return sampleMaterialTexture(tex, uv, log2(max(texelsPerPixel, 1.0)));
}

float4 textureLod(SamplerTexture2D tex, float2 uv, float lod)
{
    return sampleMaterialTexture(tex, uv, lod);
}

float4 textureGrad(SamplerTexture2D tex, float2 uv, float2 dx, float2 dy)
{
    if (tex.handle == 0u)
        return float4(1.0);
    Texture2D<float4> image = ResourceDescriptorHeap[NonUniformResourceIndex(tex.handle)];
    uint width = 0u;
    uint height = 0u;
    image.GetDimensions(width, height);
    const float2 texelDx = dx * float2(width, height);
    const float2 texelDy = dy * float2(width, height);
    return sampleMaterialTexture(tex, uv,
        0.5 * log2(max(max(dot(texelDx, texelDx), dot(texelDy, texelDy)), 1.0)));
}

int2 textureSize(SamplerTexture2D tex, int mipLevel)
{
    if (tex.handle == 0u)
        return int2(1);
    Texture2D<float4> image = ResourceDescriptorHeap[NonUniformResourceIndex(tex.handle)];
    uint width = 0u;
    uint height = 0u;
    image.GetDimensions(width, height);
    return int2(width, height);
}

// ---------------------------------------------------------------------------
// The material contract.
// ---------------------------------------------------------------------------

interface IMaterial
{
    // `parameters` is the material's parameter block, packed by the host in
    // the order the generated material reads it.
    static surfaceshader evaluate(MaterialGeometry geometry, uint* parameters);
}
