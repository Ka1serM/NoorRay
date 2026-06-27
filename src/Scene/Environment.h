#pragma once

#include <vector>
#include "Shaders/Shared.h"

class Environment {
public:
    EnvironmentSettings settings;

    static std::vector<float> computeCdf(const float* hdr, int w, int h);
};
