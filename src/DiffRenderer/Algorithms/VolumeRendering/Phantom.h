#pragma once

#include <cstdint>
#include <vector>

namespace DiffRenderer::Algorithms {

//! A synthetic head in the cube [-0.5, 0.5]^3: skull, cortex, ventricles and a tumour, as density in [0, 1] on a cubic grid.
class Phantom {
public:
    explicit Phantom(std::uint32_t size);

    std::uint32_t getSize() const { return size; }
    const std::vector<float>& getDensity() const { return density; }
    const std::vector<float>& getTumor() const { return tumor; }

private:
    std::uint32_t size = 0;
    std::vector<float> density;
    std::vector<float> tumor;
};

}
