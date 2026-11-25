#include "DiffRenderer/Training/Targets.h"

#include <algorithm>
#include <cstdint>
#include <format>

namespace DiffRenderer::Training {

using namespace NoorRHI;

namespace {

std::uint32_t toByte(const float value) {
    return static_cast<std::uint32_t>(std::clamp(value, 0.0f, 1.0f) * 255.0f + 0.5f);
}

void pack(const Scenes::View& view, std::vector<uint>& rgba) {
    const std::span<const float> values = view.image.getPixels();
    rgba.resize(values.size() / 3);
    for (std::size_t pixel = 0; pixel < rgba.size(); ++pixel)
        rgba[pixel] = toByte(values[pixel * 3]) | toByte(values[pixel * 3 + 1]) << 8 | toByte(values[pixel * 3 + 2]) << 16
                      | (view.alpha.empty() ? 255u : toByte(view.alpha[pixel])) << 24;
}

}

std::expected<void, Error> Targets::open(Device& device, const Scenes::Dataset& source) {
    dataset = &source;
    images.clear();
    images.resize(source.train.size());
    std::vector<uint> rgba;
    for (std::size_t index = 0; index < images.size(); ++index) {
        const Scenes::View& view = source.train[index];
        if (view.image.isEmpty())
            return fail(ErrorCode::InvalidArgument, std::format("the train view {} has no image", view.name));
        pack(view, rgba);
        if (std::expected<void, Error> opened = images[index].open(device, rgba.size()); !opened)
            return opened;
        if (std::expected<void, Error> uploaded = images[index].upload(rgba); !uploaded)
            return uploaded;
    }
    return {};
}

Methods::Target Targets::get(const std::size_t index) const {
    return {&dataset->train[index], static_cast<std::uint32_t>(index), images[index].getPtr()};
}

}
