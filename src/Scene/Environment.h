#pragma once

#include <vector>
#include "Scene/SceneTypes.h"

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
