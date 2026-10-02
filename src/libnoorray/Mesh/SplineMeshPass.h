#pragma once

#include <noorrhi/noorrhi.hpp>

#include <glm/vec3.hpp>

#include "Mesh/Assets/Mesh.h"
#include "Shared/SplineMesh.h"

// Bends uploaded vertex streams along a spline in place (SplineMesh.slang).
class SplineMeshPass
{
public:
    explicit SplineMeshPass(noorrhi::Device& device);

    // Makes the bent streams visible to the compute and ray-tracing work after it.
    void record(noorrhi::Buffer<glm::vec3>& positions, noorrhi::Buffer<TangentFrame>& tangents,
        const nr::graphics::SplineMesh& spline) const;

private:
    noorrhi::Device& device_;
    noorrhi::ComputePipeline pipeline_;
};
