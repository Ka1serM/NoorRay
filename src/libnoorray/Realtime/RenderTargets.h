#pragma once

#include <cstddef>

#include <noorrhi/noorrhi.hpp>

#include "Realtime/FrameContext.h"
#include "Shared/RealtimeArgs.h"

// The render-resolution images the primary pass writes and the realtime
// stages read. Named by what they hold, not by which stage reads them; see
// RenderTargetHandles for their contents.
class RenderTargets
{
public:
    RenderTargets(noorrhi::Device& device, Extent extent);

    Extent extent() const { return extent_; }
    // Storage handles for the shaders. `color` is the renderer's to choose.
    nr::graphics::RenderTargetHandles handles() const;

    noorrhi::ImageHandle diffuse() const { return diffuse_.handle(); }
    noorrhi::ImageHandle specular() const { return specular_.handle(); }
    noorrhi::ImageHandle normalRoughness() const { return normalRoughness_.handle(); }
    noorrhi::ImageHandle viewZ() const { return viewZ_.handle(); }
    noorrhi::ImageHandle motion() const { return motion_.handle(); }
    noorrhi::ImageHandle depth() const { return depth_.handle(); }
    noorrhi::ImageHandle color() const { return color_.handle(); }
    noorrhi::TextureHandle colorTexture() const { return color_.storage_handle(); }

private:
    Extent extent_;
    noorrhi::Image<std::byte> diffuse_;
    noorrhi::Image<std::byte> specular_;
    noorrhi::Image<std::byte> normalRoughness_;
    noorrhi::Image<std::byte> viewZ_;
    noorrhi::Image<std::byte> motion_;
    noorrhi::Image<std::byte> depth_;
    noorrhi::Image<std::byte> emission_;
    noorrhi::Image<std::byte> diffuseFactor_;
    noorrhi::Image<std::byte> specularFactor_;
    noorrhi::Image<std::byte> color_;
    noorrhi::Image<std::byte> albedo_;
    noorrhi::Image<std::byte> normal_;
    noorrhi::Image<std::byte> position_;
    noorrhi::Image<std::byte> cryptomatte_;
};
