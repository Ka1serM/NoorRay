#pragma once

#include <algorithm>
#include <expected>
#include <span>
#include <vector>

#include "NoorRHI/Resources/Buffer.h"
#include "NoorRay/Core/Error.h"

namespace NoorRay::Gpu {

//! Opens the buffer again when the values do not fit, with half again as much room. An empty upload keeps one element, so every pointer stays valid.
template <class T>
[[nodiscard]] std::expected<void, Core::Error> uploadValues(NoorRHI::Device& device, NoorRHI::Buffer<T>& buffer, const std::span<const T> values,
                                                            const bool growth = true) {
    const std::size_t needed = std::max<std::size_t>(values.size(), 1);
    if (!buffer || buffer.getSize() < needed || (!growth && buffer.getSize() != needed)) {
        const std::size_t count = growth ? needed + needed / 2 : needed;
        if (auto opened = buffer.open(device, count); !opened)
            return Core::fail(std::move(opened.error()));
    }
    return Core::check(buffer.upload(values));
}

template <class T>
[[nodiscard]] std::expected<void, Core::Error> openZeroed(NoorRHI::Device& device, NoorRHI::Buffer<T>& buffer, const std::size_t count) {
    if (auto opened = buffer.open(device, count); !opened)
        return Core::fail(std::move(opened.error()));
    const std::size_t chunk = std::clamp<std::size_t>((std::size_t{1} << 20) / sizeof(T), 1, count);
    const std::vector<T> zeros(chunk);
    for (std::size_t offset = 0; offset < count; offset += chunk)
        if (auto uploaded = buffer.upload(std::span<const T>(zeros).first(std::min(chunk, count - offset)), offset); !uploaded)
            return Core::fail(std::move(uploaded.error()));
    return {};
}

}
