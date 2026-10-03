#include "NrdSignal.slang"
#include "RealtimeCamera.slang"

static const uint RealtimeCompositeGroupSize = 8u;

// Upsampling rejects lighting samples whose view Z differs from the render
// pixel's by more than this fraction of it (widened at grazing angles, as
// NRD's re-jittering does), or whose normal turns away.
static const float UpsampleDepthSigma = 0.03;
static const float UpsampleNormalPower = 8.0;
// Weight of the depth-closest sample regardless of the others, so a pixel no
// sample matches fades into it instead of switching to it.
static const float UpsampleFallbackWeight = 1.0e-3;

// The denoised lobes at one render pixel, still packed.
struct LightingSample
{
    float4 diffuse;
    float4 specular;
    float4 diffuseSh1;
    float4 specularSh1;
};

// One texel of an SH1 image. They exist only in SH form, so only its
// branches fetch their descriptors.
float4 loadSh1(uint image, int2 pixel)
{
    RWTexture2D<float4> sh1 = ResourceDescriptorHeap[image];
    return sh1[pixel];
}

// Joint bilateral upsampling of the denoised lighting onto a render pixel:
// the lighting samples around it, weighted by a tent over the lighting grid
// and by how well their surface matches the render pixel's. The tent spans one
// lighting pixel at half resolution (bilinear) and two below, so coarse
// lighting still blends enough samples to stay stable in motion. The packed
// signals are linear in radiance, so they blend directly. At full lighting
// resolution the render pixel is its own lighting sample.
LightingSample upsampleLighting(uint2 pixel, float viewZ, float3 normal, float normalView,
    uint diffuseSource, uint specularSource, uint diffuseShSource, uint specularShSource,
    uint viewZSource, uint normalSource, bool sphericalHarmonics)
{
    RealtimeArgs* args = realtimeArgs();
    RWTexture2D<float4> diffuse = ResourceDescriptorHeap[diffuseSource];
    RWTexture2D<float4> specular = ResourceDescriptorHeap[specularSource];
    const float scale = args.view.lightingScale;
    LightingSample result = (LightingSample)0;
    if (scale == 1.0f)
    {
        result.diffuse = diffuse[pixel];
        result.specular = specular[pixel];
        if (sphericalHarmonics)
        {
            result.diffuseSh1 = loadSh1(diffuseShSource, int2(pixel));
            result.specularSh1 = loadSh1(specularShSource, int2(pixel));
        }
        return result;
    }
    RWTexture2D<float4> lightingViewZ = ResourceDescriptorHeap[viewZSource];
    RWTexture2D<float4> lightingNormalRoughness = ResourceDescriptorHeap[normalSource];

    // Render and lighting samples are jittered by the same screen distance,
    // so the jitter cancels: render pixel p lies at (p + 0.5) / scale - 0.5 on
    // the grid of lighting samples.
    const float2 position = (float2(pixel) + 0.5) / float(scale) - 0.5;
    const int radius = scale <= 2.0f ? 1 : 2;
    const int2 base = int2(floor(position));
    const int2 lastPixel = int2(args.view.lightingWidth, args.view.lightingHeight) - 1;
    const float depthTolerance = UpsampleDepthSigma / (abs(normalView) * 0.95 + 0.05);

    float weightSum = 0.0;
    int2 nearest = clamp(int2(round(position)), int2(0), lastPixel);
    float nearestDepthError = 1.0e30;
    for (int y = 1 - radius; y <= radius; ++y)
    {
        for (int x = 1 - radius; x <= radius; ++x)
        {
            const int2 source = clamp(base + int2(x, y), int2(0), lastPixel);
            const float2 distance = abs(float2(base + int2(x, y)) - position);
            const float2 tent = saturate(1.0 - distance / float(radius));
            const float depthError = abs(lightingViewZ[source].x - viewZ) / max(viewZ, 1.0e-4);
            if (depthError < nearestDepthError)
            {
                nearestDepthError = depthError;
                nearest = source;
            }
            const float3 sourceNormal =
                NRD_FrontEnd_UnpackNormalAndRoughness(lightingNormalRoughness[source]).xyz;
            const float weight = tent.x * tent.y * exp(-depthError / depthTolerance)
                * pow(saturate(dot(sourceNormal, normal)), UpsampleNormalPower);
            if (!(weight > 0.0))
                continue;
            weightSum += weight;
            result.diffuse += diffuse[source] * weight;
            result.specular += specular[source] * weight;
            if (sphericalHarmonics)
            {
                result.diffuseSh1 += loadSh1(diffuseShSource, source) * weight;
                result.specularSh1 += loadSh1(specularShSource, source) * weight;
            }
        }
    }
    weightSum += UpsampleFallbackWeight;
    result.diffuse += diffuse[nearest] * UpsampleFallbackWeight;
    result.specular += specular[nearest] * UpsampleFallbackWeight;
    if (sphericalHarmonics)
    {
        result.diffuseSh1 += loadSh1(diffuseShSource, nearest) * UpsampleFallbackWeight;
        result.specularSh1 += loadSh1(specularShSource, nearest) * UpsampleFallbackWeight;
    }
    const float normalization = 1.0 / weightSum;
    result.diffuse *= normalization;
    result.specular *= normalization;
    result.diffuseSh1 *= normalization;
    result.specularSh1 *= normalization;
    return result;
}

// NRD's resolve of the SH lobes at a render pixel (see the NRD sample's
// Composition.cs.hlsl): each lobe's spherical Gaussian resolved against the
// pixel's own normal and roughness regains macro detail, and re-jittering,
// which weighs the pixel's BRDF against its four neighbours', regains micro
// detail and the per-pixel jitter the upscaler relies on.
void resolveSh(LightingSample lighting, uint2 pixel, float3 normal, float3 view,
    float roughness, float viewZ, out float3 diffuse, out float3 specular)
{
    RealtimeArgs* args = realtimeArgs();
    RWTexture2D<float4> normalRoughness = ResourceDescriptorHeap[args.targets.normalRoughness];
    RWTexture2D<float4> viewZs = ResourceDescriptorHeap[args.targets.viewZ];
    const NRD_SG diffuseSg = unpackNrdSh(lighting.diffuse, lighting.diffuseSh1);
    const NRD_SG specularSg = unpackNrdSh(lighting.specular, lighting.specularSh1);
    diffuse = NRD_SG_ResolveDiffuse(diffuseSg, normal, view, roughness);
    specular = NRD_SG_ResolveSpecular(specularSg, normal, view, roughness);

    const int2 lastPixel = int2(args.frame.width, args.frame.height) - 1;
    const int2 east = clamp(int2(pixel) + int2(1, 0), int2(0), lastPixel);
    const int2 west = clamp(int2(pixel) + int2(-1, 0), int2(0), lastPixel);
    const int2 north = clamp(int2(pixel) + int2(0, 1), int2(0), lastPixel);
    const int2 south = clamp(int2(pixel) + int2(0, -1), int2(0), lastPixel);
    const float2 rejitter = NRD_SG_ReJitter(diffuseSg, specularSg, view, roughness, viewZ,
        viewZs[east].x, viewZs[west].x, viewZs[north].x, viewZs[south].x, normal,
        NRD_FrontEnd_UnpackNormalAndRoughness(normalRoughness[east]).xyz,
        NRD_FrontEnd_UnpackNormalAndRoughness(normalRoughness[west]).xyz,
        NRD_FrontEnd_UnpackNormalAndRoughness(normalRoughness[north]).xyz,
        NRD_FrontEnd_UnpackNormalAndRoughness(normalRoughness[south]).xyz);
    diffuse *= rejitter.x;
    specular *= rejitter.y;
}

// Recombines the denoiser's diffuse and specular radiance with the material
// factors they were demodulated by, adds the radiance that bypassed the
// denoiser, and writes the render-resolution color target. The lighting is
// upsampled from the lighting resolution and, from the SH denoisers,
// resolved against each render pixel's own surface.
[shader("compute")]
[numthreads(RealtimeCompositeGroupSize, RealtimeCompositeGroupSize, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    RealtimeArgs* args = realtimeArgs();
    if (threadId.x >= args.frame.width || threadId.y >= args.frame.height)
        return;
    const uint2 pixel = threadId.xy;

    RWTexture2D<float4> color = ResourceDescriptorHeap[args.targets.color];
    RWTexture2D<float4> emission = ResourceDescriptorHeap[args.targets.emission];
    RWTexture2D<float4> viewZ = ResourceDescriptorHeap[args.targets.viewZ];
    RWTexture2D<float4> normalRoughness = ResourceDescriptorHeap[args.targets.normalRoughness];
    RWTexture2D<float4> diffuseFactor = ResourceDescriptorHeap[args.targets.diffuseFactor];
    RWTexture2D<float4> specularFactor = ResourceDescriptorHeap[args.targets.specularFactor];

    const float4 base = emission[pixel];
    float3 radiance = base.rgb;
    const float pixelViewZ = viewZ[pixel].x;
    // Background carries no lighting to resolve.
    if (pixelViewZ < args.denoiser.denoisingRange)
    {
        const float4 surface = NRD_FrontEnd_UnpackNormalAndRoughness(normalRoughness[pixel]);
        const float3 view = -realtimeCameraRay(pixel).direction;
        const LightingSample lighting = upsampleLighting(pixel, pixelViewZ, surface.xyz,
            dot(surface.xyz, view), args.denoiser.diffuse, args.denoiser.specular,
            args.denoiser.diffuseSh1, args.denoiser.specularSh1,
            args.targets.lightingViewZ, args.targets.lightingNormalRoughness,
            args.denoiser.sphericalHarmonics != 0u);
        float3 diffuse;
        float3 specular;
        if (args.denoiser.sphericalHarmonics != 0u)
        {
            resolveSh(lighting, pixel, surface.xyz, view, surface.w, pixelViewZ, diffuse,
                specular);
        }
        else
        {
            diffuse = unpackNrdRadiance(lighting.diffuse);
            specular = unpackNrdRadiance(lighting.specular);
        }
        radiance += diffuse * diffuseFactor[pixel].rgb + specular * specularFactor[pixel].rgb;
    }
#if defined(REALTIME_LAYERED)
    RWTexture2D<float4> layerViewZ = ResourceDescriptorHeap[args.targets.layerViewZ];
    const float layerDepth = layerViewZ[pixel].x;
    if (layerDepth < args.denoiser.denoisingRange)
    {
        RWTexture2D<float4> layerNormal = ResourceDescriptorHeap[args.targets.layerNormalRoughness];
        RWTexture2D<float4> layerDiffuseFactor = ResourceDescriptorHeap[args.targets.layerDiffuseFactor];
        RWTexture2D<float4> layerSpecularFactor = ResourceDescriptorHeap[args.targets.layerSpecularFactor];
        const float3 surfaceNormal = NRD_FrontEnd_UnpackNormalAndRoughness(layerNormal[pixel]).xyz;
        const LightingSample layer = upsampleLighting(pixel, layerDepth, surfaceNormal, 1.0,
            args.denoiser.layerDiffuse, args.denoiser.layerSpecular,
            args.denoiser.diffuseSh1, args.denoiser.specularSh1,
            args.targets.layerLightingViewZ, args.targets.layerLightingNormalRoughness, false);
        radiance += unpackNrdRadiance(layer.diffuse) * layerDiffuseFactor[pixel].rgb
            + unpackNrdRadiance(layer.specular) * layerSpecularFactor[pixel].rgb;
    }
#endif
    if (any(isnan(radiance)) || any(isinf(radiance)))
        radiance = float3(0.0);
    color[pixel] = float4(clamp(radiance, 0.0, RealtimeHalfMax), base.a);
}
