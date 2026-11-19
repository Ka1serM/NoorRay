#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <unordered_map>
#include <vector>

#include "NoorRHI/Resources/Image.h"
#include "NoorRHI/Resources/Sampler.h"
#include "NoorRay/Core/Error.h"
#include "NoorRay/Scene/Scene.h"

namespace NoorRay::Gpu {

//! One sampled image per scene texture, kept by handle so that adding a texture uploads only that texture, and one sampler per wrap and filter mode.
class TextureTable {
public:
    [[nodiscard]] std::expected<void, Core::Error> open(NoorRHI::Device& device);
    [[nodiscard]] std::expected<void, Core::Error> update(NoorRHI::Device& device, const Scene::Scene& scene);

    NoorRHI::TextureHandle getTexture(Scene::TextureHandle handle) const;
    NoorRHI::SamplerHandle getSampler(Scene::TextureWrap wrap, Scene::TextureFilter filter) const;
    std::size_t getByteSize() const;

private:
    static std::uint64_t getKey(const Scene::TextureHandle handle) { return std::uint64_t{handle.generation} << 32 | handle.index; }

    std::unordered_map<std::uint64_t, NoorRHI::Image> images;
    std::array<NoorRHI::Sampler, 8> samplers;
};

}
