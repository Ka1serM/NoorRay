#pragma once

#include <cstddef>

#include <noorrhi/noorrhi.hpp>

#include "Realtime/FrameContext.h"
#include "Shared/RealtimeArgs.h"

// What RenderTargets allocates: the extents it holds, which frames of any
// smaller rectangle render into, and the targets only some configurations
// read.
struct RenderTargetLayout
{
    Extent render;
    Extent lighting;
    // Lighting runs at the render resolution, so the lighting samples' guides
    // are the G-buffer's own.
    bool sharedGuides{};
    // Translucent layers: their lighting, guides and factors, and the
    // upscaler's transparency mask.
    bool layers{};
    bool sphericalHarmonics{};
    // The composite writes a color target for the upscaler instead of the
    // output image.
    bool upscaled{};

    bool operator==(const RenderTargetLayout&) const = default;
};

// The images the primary passes write and the realtime stages read: the
// G-buffer at the render resolution, and the lighting samples with their
// denoiser guides at the lighting resolution. Named by what they hold, not by
// which stage reads them; see RenderTargetHandles for their contents. Targets
// the layout leaves out are empty, with a zero handle no pass reads.
class RenderTargets
{
public:
    RenderTargets(noorrhi::Device& device, const RenderTargetLayout& layout);

    const RenderTargetLayout& layout() const { return layout_; }
    // Storage handles for the shaders. `color` is the renderer's to choose.
    nr::graphics::RenderTargetHandles handles() const;

    noorrhi::ImageHandle diffuse() const { return diffuse_.handle(); }
    noorrhi::ImageHandle specular() const { return specular_.handle(); }
    noorrhi::ImageHandle diffuseSh1() const { return diffuseSh1_.handle(); }
    noorrhi::ImageHandle specularSh1() const { return specularSh1_.handle(); }
    noorrhi::ImageHandle lightingNormalRoughness() const { return lightingNormalRoughnessImage().handle(); }
    noorrhi::ImageHandle lightingViewZ() const { return lightingViewZImage().handle(); }
    noorrhi::ImageHandle lightingMotion() const { return lightingMotionImage().handle(); }
    noorrhi::ImageHandle layerDiffuse() const { return layerDiffuse_.handle(); }
    noorrhi::ImageHandle layerSpecular() const { return layerSpecular_.handle(); }
    noorrhi::ImageHandle layerLightingNormalRoughness() const { return layerLightingNormalRoughnessImage().handle(); }
    noorrhi::ImageHandle layerLightingViewZ() const { return layerLightingViewZImage().handle(); }
    noorrhi::ImageHandle layerLightingMotion() const { return layerLightingMotion_.handle(); }
    noorrhi::ImageHandle transparencyMask() const { return transparencyMask_.handle(); }
    noorrhi::ImageHandle motion() const { return motion_.handle(); }
    noorrhi::ImageHandle depth() const { return depth_.handle(); }
    noorrhi::ImageHandle color() const { return color_.handle(); }
    noorrhi::TextureHandle colorTexture() const { return color_.storage_handle(); }
    const noorrhi::Image<std::byte>& cryptomatteImage() const { return cryptomatte_; }

private:
    const noorrhi::Image<std::byte>& lightingNormalRoughnessImage() const
    {
        return layout_.sharedGuides ? normalRoughness_ : lightingNormalRoughness_;
    }
    const noorrhi::Image<std::byte>& lightingViewZImage() const
    {
        return layout_.sharedGuides ? viewZ_ : lightingViewZ_;
    }
    const noorrhi::Image<std::byte>& lightingMotionImage() const
    {
        return layout_.sharedGuides ? motion_ : lightingMotion_;
    }
    const noorrhi::Image<std::byte>& layerLightingNormalRoughnessImage() const
    {
        return layout_.sharedGuides ? layerNormalRoughness_ : layerLightingNormalRoughness_;
    }
    const noorrhi::Image<std::byte>& layerLightingViewZImage() const
    {
        return layout_.sharedGuides ? layerViewZ_ : layerLightingViewZ_;
    }

    RenderTargetLayout layout_;
    noorrhi::Image<std::byte> diffuse_;
    noorrhi::Image<std::byte> specular_;
    noorrhi::Image<std::byte> diffuseSh1_;
    noorrhi::Image<std::byte> specularSh1_;
    noorrhi::Image<std::byte> lightingNormalRoughness_;
    noorrhi::Image<std::byte> lightingViewZ_;
    noorrhi::Image<std::byte> lightingMotion_;
    noorrhi::Image<std::byte> layerDiffuse_;
    noorrhi::Image<std::byte> layerSpecular_;
    noorrhi::Image<std::byte> layerLightingNormalRoughness_;
    noorrhi::Image<std::byte> layerLightingViewZ_;
    noorrhi::Image<std::byte> layerLightingMotion_;
    noorrhi::Image<std::byte> normalRoughness_;
    noorrhi::Image<std::byte> viewZ_;
    noorrhi::Image<std::byte> motion_;
    noorrhi::Image<std::byte> depth_;
    noorrhi::Image<std::byte> emission_;
    noorrhi::Image<std::byte> diffuseFactor_;
    noorrhi::Image<std::byte> specularFactor_;
    noorrhi::Image<std::byte> layerDiffuseFactor_;
    noorrhi::Image<std::byte> layerSpecularFactor_;
    noorrhi::Image<std::byte> layerNormalRoughness_;
    noorrhi::Image<std::byte> layerViewZ_;
    noorrhi::Image<std::byte> transparencyMask_;
    noorrhi::Image<std::byte> color_;
    noorrhi::Image<std::byte> cryptomatte_;
};
