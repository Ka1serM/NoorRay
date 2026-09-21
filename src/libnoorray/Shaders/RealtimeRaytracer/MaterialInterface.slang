// The contract between the realtime renderer and the materials it shades.
//
// MaterialX documents are generated into Slang modules against this module
// (Materials/MaterialX/SlangMaterialGenerator.cpp), which the runtime links
// into one callable shader per distinct material. The realtime closest hit
// calls a material's callable with a MaterialCall and shades the closures
// the material returns with its own GGX model.
//
// MaterialX's closure types are defined here, not by MaterialX's hardware
// libraries: a BSDF is not evaluated against lights inside the material but
// returned as the lobes the renderer samples. Every BSDF reduces to a
// Lambertian diffuse lobe plus one GGX reflection lobe. Closure operations
// combine lobes linearly; each lobe's roughness and normal are averaged by
// the weight the lobe contributes.

// Where a material is evaluated. World space; directions are normalized.
struct MaterialGeometry
{
    float3 position;
    float3 normal;
    float3 tangent;
    float3 bitangent;
    float2 uv;
    float4 color;
};

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
// Textures. MaterialX filename parameters are resource-descriptor-heap
// indices; 0 means no texture and samples white.
// ---------------------------------------------------------------------------

struct SamplerTexture2D
{
    uint handle;
}

// Bilinear, wrapped addressing (MaterialX applies its own address modes to
// the coordinates first). Images are single-level, so there is no mip
// selection; sRGB and block-compressed formats are decoded by the image view.
float4 texture(SamplerTexture2D tex, float2 uv)
{
    if (tex.handle == 0u)
        return float4(1.0);
    Texture2D<float4> image = ResourceDescriptorHeap[tex.handle];
    uint width = 0u;
    uint height = 0u;
    image.GetDimensions(width, height);
    if (width == 0u || height == 0u)
        return float4(1.0);

    const int2 size = int2(width, height);
    const float2 p = frac(uv) * float2(size) - 0.5;
    const int2 base = int2(floor(p));
    const float2 f = p - float2(base);
    const int2 p00 = (base + size) % size;
    const int2 p11 = (base + 1 + size) % size;
    const float4 x0 = lerp(image.Load(int3(p00, 0)),
        image.Load(int3(int2(p11.x, p00.y), 0)), f.x);
    const float4 x1 = lerp(image.Load(int3(int2(p00.x, p11.y), 0)),
        image.Load(int3(p11, 0)), f.x);
    return lerp(x0, x1, f.y);
}

float4 textureLod(SamplerTexture2D tex, float2 uv, float lod)
{
    return texture(tex, uv);
}

float4 textureGrad(SamplerTexture2D tex, float2 uv, float2 dx, float2 dy)
{
    return texture(tex, uv);
}

int2 textureSize(SamplerTexture2D tex, int mipLevel)
{
    if (tex.handle == 0u)
        return int2(1);
    Texture2D<float4> image = ResourceDescriptorHeap[tex.handle];
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

// The callable data of a material call. The parameter block is passed as a
// plain address: NVIDIA's driver crashes compiling pipelines whose callable
// data holds a pointer.
struct MaterialCall
{
    MaterialGeometry geometry;
    uint64_t parameters;
    surfaceshader surface;
}

void callMaterial<M : IMaterial>(inout MaterialCall call)
{
    call.surface = M.evaluate(call.geometry, (uint*)call.parameters);
}
