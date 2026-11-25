#include "NoorRay/Gpu/TextureTable.h"

#include <algorithm>
#include <bit>
#include <unordered_set>

#include "NoorRHI/Device/Device.h"

namespace NoorRay::Gpu {

namespace {

NoorRHI::AddressMode toAddressMode(const Scene::TextureWrap wrap, const bool hasBorder) {
    switch (wrap) {
    case Scene::TextureWrap::Extend:
        return NoorRHI::AddressMode::ClampToEdge;
    case Scene::TextureWrap::Clip:
        return hasBorder ? NoorRHI::AddressMode::ClampToBorder : NoorRHI::AddressMode::ClampToEdge;
    case Scene::TextureWrap::Mirror:
        return NoorRHI::AddressMode::MirroredRepeat;
    default:
        return NoorRHI::AddressMode::Repeat;
    }
}

std::size_t getSamplerIndex(const Scene::TextureWrap wrap, const Scene::TextureFilter filter) {
    const std::size_t nearest = filter == Scene::TextureFilter::Closest ? 1 : 0;
    return static_cast<std::size_t>(wrap) * 2 + nearest;
}

}

std::expected<void, Core::Error> TextureTable::open(NoorRHI::Device& device) {
    //! WebGPU has no border colour, so Clip falls back to the edge texel there.
    const bool hasBorder = device.getInfo().api != NoorRHI::GraphicsApi::WebGpu;
    for (std::uint8_t wrap = 0; wrap < 4; ++wrap)
        for (std::uint8_t nearest = 0; nearest < 2; ++nearest) {
            const NoorRHI::AddressMode mode = toAddressMode(static_cast<Scene::TextureWrap>(wrap), hasBorder);
            const NoorRHI::SamplerDesc desc{.filter = nearest ? NoorRHI::Filter::Nearest : NoorRHI::Filter::Linear,
                                            .addressU = mode,
                                            .addressV = mode,
                                            .addressW = mode,
                                            .maxAnisotropy = nearest ? 1.0f : 8.0f};
            if (auto opened = samplers[wrap * 2 + nearest].open(device, desc); !opened)
                return Core::fail(std::move(opened.error()));
        }
    return {};
}

std::expected<void, Core::Error> TextureTable::update(NoorRHI::Device& device, const Scene::Scene& scene) {
    const Core::SlotMap<Scene::Texture>& textures = scene.getTextures();
    std::unordered_set<std::uint64_t> live;
    for (std::uint32_t dense = 0; dense < textures.size(); ++dense) {
        const std::uint64_t key = getKey(textures.keyAt(dense));
        live.insert(key);
        if (images.contains(key))
            continue;
        const Scene::Texture& texture = textures.getValues()[dense];
        const NoorRHI::ImageDesc desc{.width = texture.width,
                                      .height = texture.height,
                                      .format = texture.format,
                                      .usage = NoorRHI::ImageUsage::Sampled,
                                      .mipLevels = static_cast<std::uint32_t>(std::bit_width(std::max(texture.width, texture.height))),
                                      .name = texture.name};
        NoorRHI::Image image;
        if (auto opened = image.open(device, desc); !opened)
            return Core::fail(Core::ErrorCode::Gpu, "texture '" + texture.name + "': " + opened.error().message);
        if (auto uploaded = image.upload(texture.data); !uploaded)
            return Core::fail(Core::ErrorCode::Gpu, "texture '" + texture.name + "': " + uploaded.error().message);
        images.emplace(key, std::move(image));
    }
    std::erase_if(images, [&](const auto& entry) { return !live.contains(entry.first); });
    return {};
}

NoorRHI::TextureHandle TextureTable::getTexture(const Scene::TextureHandle handle) const {
    const auto found = images.find(getKey(handle));
    return found == images.end() ? NoorRHI::TextureHandle{} : found->second.getSampledHandle();
}

NoorRHI::SamplerHandle TextureTable::getSampler(const Scene::TextureWrap wrap, const Scene::TextureFilter filter) const {
    return samplers[getSamplerIndex(wrap, filter)].getHandle();
}

std::size_t TextureTable::getByteSize() const {
    std::size_t bytes = 0;
    for (const auto& [key, image] : images)
        bytes += image.getByteSize();
    return bytes;
}

}
