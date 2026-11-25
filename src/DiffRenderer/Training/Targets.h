#pragma once

#include <cstddef>
#include <expected>
#include <vector>

#include "DiffRenderer/Methods/Method.h"
#include "DiffRenderer/Scenes/Dataset.h"
#include "NoorRHI/NoorRHI.h"

namespace DiffRenderer::Training {

//! The train images of a dataset on the GPU as RGBA8, in the encoding of the dataset; the alpha byte holds the mask, or 255 without one.
class Targets {
public:
    [[nodiscard]] std::expected<void, NoorRHI::Error> open(NoorRHI::Device& device, const Scenes::Dataset& dataset);
    Methods::Target get(std::size_t index) const;
    std::size_t getCount() const { return images.size(); }

private:
    const Scenes::Dataset* dataset = nullptr;
    std::vector<NoorRHI::Buffer<NoorRHI::uint>> images;
};

}
