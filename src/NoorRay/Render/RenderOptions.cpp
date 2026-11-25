#include "NoorRay/Render/RenderOptions.h"

namespace NoorRay::Render {

RendererSettings RenderOptions::toRendererSettings() const {
    RendererSettings renderer = RendererSettings::pathTraced();
    renderer.maxSamples = samples;
    renderer.denoiser = denoiser;
    renderer.temporal = temporal;
    //! The viewport renders each frame on its own, as in a game; a final render accumulates up to samples.
    renderer.accumulate = finalRender;
    renderer.upscaler = temporal && upscale ? Upscaler::Fsr : Upscaler::Off;
    renderer.renderScale = renderScale;
    renderer.sharpness = sharpness;
    renderer.reuse = reuse;
    renderer.radianceCache = radianceCache;
    renderer.wavefront.lightTree = lightTree;
    renderer.wavefront.visibilityCache = visibilityCache;
    renderer.wavefront.gaussianShading = gaussianShading;
    renderer.wavefront.seed = seed;
    renderer.wavefront.compilesInBackground = !finalRender;
    if (denoiser == Denoiser::OpenDlssRr) {
        //! DLSS-RR needs temporal frames (RendererSettings::validated); it replaces RELAX and FSR.
        renderer.temporal = true;
        renderer.upscaler = Upscaler::Off;
    }
    //! DLSS-RR does not accumulate, so RELAX denoises a final render.
    if (finalRender && renderer.denoiser == Denoiser::OpenDlssRr)
        renderer.denoiser = Denoiser::Nrd;
    renderer.wavefront.maxDepth = maxBounces + 1;
    renderer.wavefront.maxVolumeBounces = volumeBounces;
    renderer.wavefront.maxTransparentBounces = transparentBounces;
    renderer.wavefront.maxDiffuseBounces = diffuseBounces;
    renderer.wavefront.maxGlossyBounces = glossyBounces;
    renderer.wavefront.maxTransmissionBounces = transmissionBounces;
    renderer.wavefront.filterGlossy = filterGlossy;
    renderer.wavefront.exposure = exposure;
    renderer.wavefront.directClamp = clampDirect;
    renderer.wavefront.indirectClamp = clampIndirect;
    renderer.wavefront.filterWidth = filterWidth;
    return renderer;
}

}
