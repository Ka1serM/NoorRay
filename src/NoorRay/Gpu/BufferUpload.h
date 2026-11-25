#pragma once

#include <algorithm>
#include <expected>
#include <span>

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
    return Core::check(buffer.clear());
}

}
