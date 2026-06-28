#pragma once

#include <cstdint>

#include <glm/vec2.hpp>

#include "GPU/Annotations.h"

class CameraInstance;

struct Sensor {
    float widthMm{5.784f};
    float heightMm{3.264f};
    uint32_t resolutionWidth{1928};
    uint32_t resolutionHeight{1088};

    NR_CPU_GPU glm::uvec2 resolution() const { return {resolutionWidth, resolutionHeight}; }
    NR_CPU_GPU float aspectRatio() const { return widthMm / heightMm; }

    void renderUi(CameraInstance& instance);
};
