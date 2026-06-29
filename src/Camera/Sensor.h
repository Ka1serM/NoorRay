#pragma once

#include <algorithm>
#include <cstdint>

#include <glm/vec2.hpp>

#include "CUDA/Annotations.h"

struct Sensor {
    float widthMm{5.784f};
    float heightMm{3.264f};
    uint32_t resolutionWidth{1928};
    uint32_t resolutionHeight{1088};

    NR_CPU_GPU glm::uvec2 resolution() const { return {resolutionWidth, resolutionHeight}; }
    NR_CPU_GPU void setResolution(uint32_t w, uint32_t h) { resolutionWidth = std::max(1u, w); resolutionHeight = std::max(1u, h); }
    NR_CPU_GPU float aspectRatio() const { return widthMm / heightMm; }

    bool renderUi();
};
