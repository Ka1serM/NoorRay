#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "NoorRay.Python/Blender/Reader.h"
#include "NoorRay/Scene/Scene.h"

namespace NoorRay::Python {

//! Turns the hair curves, particle hair and points of Blender into meshes: tubes for strands and spheres for points.
class StrandReader {
public:
    explicit StrandReader(Reader& reader) : reader(reader) {}

    //! The hair of a Curves data-block; a mesh without triangles has no indices.
    Scene::Mesh readCurves(const Object& curves, Scene::MaterialHandle material);
    Scene::Mesh readPoints(const Object& points, Scene::MaterialHandle material);
    //! One mesh for each particle system that renders hair as paths.
    std::vector<Scene::Mesh> readParticles(const Object& object, bool render, std::span<const Scene::MaterialHandle> slotMaterials);

private:
    std::vector<float> attribute(const Object& data, const char* name, const char* field, std::size_t components, std::size_t count);

    Reader& reader;
};

//! The share of the cells over the bounds that a closed mesh fills, as halves; the parity of the crossings along z tells the inside.
std::vector<std::uint16_t> fillMask(std::span<const Core::float3> positions, std::span<const std::uint32_t> indices, const Core::Bounds& bounds,
                                    Core::uint3 resolution);

}
