#pragma once

#include <noorrhi/noorrhi.hpp>
#include <glm/vec3.hpp>
#include "Shared/LandscapeMesh.h"

// Converts original landscape height texels into positions before BLAS building.
class LandscapePass
{
public:
    explicit LandscapePass(noorrhi::Device& device);
    void record(noorrhi::Buffer<glm::vec3>& positions) const;
private:
    noorrhi::Device& device_;
    noorrhi::ComputePipeline pipeline_;
};
