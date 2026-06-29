#pragma once

#include <cstdint>
#include <vector>

#include <cuda_runtime_api.h>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
using glm::vec3;

#include "CUDA/Annotations.h"
#include "CUDA/Texture.h"

class Environment
{
public:
    int textureIndex{-1};
    vec3 color{1.0f};
    float rotationSin{};
    float rotationCos{1.0f};
    float visibleExposureScale{1.0f};
    float lightingExposureScale{1.0f};
    int visible{1};
    float rotation{};
    float visibleExposure{};
    float lightingExposure{1.0f};
    cudaArray_t cdfArray{};
    cudaTextureObject_t cdfTexture{};

    Environment();
    ~Environment();
    Environment(const Environment&) = delete;
    Environment& operator=(const Environment&) = delete;

    void destroyCdf() noexcept;
    void updateDerivedSettings();

    static std::vector<float> computeCdf(const float* hdr, int w, int h);
};
