#pragma once

#include <vector>

#include <glm/vec3.hpp>
using glm::vec3;

struct EnvironmentSettings
{
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
};

class Environment {
public:
    EnvironmentSettings* settings{};

    Environment();
    ~Environment();
    Environment(const Environment&) = delete;
    Environment& operator=(const Environment&) = delete;

    void updateDerivedSettings();

    static std::vector<float> computeCdf(const float* hdr, int w, int h);
};
