#include "NrdSignal.slang"

static const uint RealtimeCompositeGroupSize = 8u;

// Recombines the denoiser's diffuse and specular radiance with the material
// factors they were demodulated by, adds the radiance that bypassed the
// denoiser, and writes the render-resolution color target.
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
    RWTexture2D<float4> diffuseFactor = ResourceDescriptorHeap[args.targets.diffuseFactor];
    RWTexture2D<float4> specularFactor = ResourceDescriptorHeap[args.targets.specularFactor];
    RWTexture2D<float4> diffuse = ResourceDescriptorHeap[args.denoiser.diffuse];
    RWTexture2D<float4> specular = ResourceDescriptorHeap[args.denoiser.specular];

    const float4 base = emission[pixel];
    float3 radiance = base.rgb;
    // NRD leaves pixels beyond its denoising range untouched.
    if (viewZ[pixel].x < args.denoiser.denoisingRange)
        radiance += unpackNrdRadiance(diffuse[pixel]) * diffuseFactor[pixel].rgb
            + unpackNrdRadiance(specular[pixel]) * specularFactor[pixel].rgb;
    if (any(isnan(radiance)) || any(isinf(radiance)))
        radiance = float3(0.0);
    color[pixel] = float4(max(radiance, 0.0) * exp2(args.frame.camera.exposure), base.a);
}
