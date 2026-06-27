#pragma once

#include <vector>
#include "Scene/SceneTypes.h"

class Environment {
public:
    EnvironmentSettings settings;

    static std::vector<float> computeCdf(const float* hdr, int w, int h);
};
