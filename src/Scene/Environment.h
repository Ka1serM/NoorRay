#pragma once

#include <vector>

#include "Scene/SceneTypes.h"

struct EnvironmentSettings
{
    int textureIndex{-1};
    int cdfTextureIndex{-1};
    float rotationSin{};
    float rotationCos{1.0f};
    float visibleExposureScale{1.0f};
    float lightingExposureScale{1.0f};
    float maxTextureLod{};
    int visible{1};
    vec3 directionalDirection{0.0f, 1.0f, 0.0f};
    float directionalIntensity{};
    float rotation{};
    float visibleExposure{};
    float lightingExposure{1.0f};
    float directionalSoftAngle{};
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
