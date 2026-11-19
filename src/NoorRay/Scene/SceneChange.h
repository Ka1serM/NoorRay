#pragma once

#include <cstdint>

namespace NoorRay::Scene {

enum class Change : std::uint32_t {
    None = 0,
    InstanceTransforms = 1u << 0,
    InstanceSet = 1u << 1,
    MeshPositions = 1u << 2,
    MeshTopology = 1u << 3,
    MaterialValues = 1u << 4,
    MaterialSet = 1u << 5,
    Textures = 1u << 6,
    Lights = 1u << 7,
    GaussianValues = 1u << 8,
    GaussianSet = 1u << 9,
    Volumes = 1u << 10,
    Camera = 1u << 11,
    Environment = 1u << 12,
};

constexpr Change operator|(const Change a, const Change b) {
    return static_cast<Change>(static_cast<std::uint32_t>(a) | static_cast<std::uint32_t>(b));
}
constexpr Change operator&(const Change a, const Change b) {
    return static_cast<Change>(static_cast<std::uint32_t>(a) & static_cast<std::uint32_t>(b));
}
constexpr Change& operator|=(Change& a, const Change b) {
    return a = a | b;
}

inline constexpr Change geometryChanges =
    Change::InstanceTransforms | Change::InstanceSet | Change::MeshPositions | Change::MeshTopology | Change::GaussianValues | Change::GaussianSet;
//! Changes that make cached path radiance stale, so ReSTIR reservoirs need validation.
inline constexpr Change lightingChanges =
    geometryChanges | Change::MaterialValues | Change::MaterialSet | Change::Textures | Change::Lights | Change::Volumes | Change::Environment;

//! The changes since the last GpuScene update, with the dirty dense index range of each kind that only needs a range upload.
class SceneChange {
public:
    struct Range {
        std::uint32_t begin = UINT32_MAX;
        std::uint32_t end = 0;

        void add(const std::uint32_t index) {
            begin = index < begin ? index : begin;
            end = index + 1 > end ? index + 1 : end;
        }
        bool isEmpty() const { return begin >= end; }
    };

    void mark(const Change change) { changes |= change; }
    void markInstance(const std::uint32_t index) {
        changes |= Change::InstanceTransforms;
        instances.add(index);
    }
    void markMesh(const std::uint32_t index) {
        changes |= Change::MeshPositions;
        meshes.add(index);
    }
    void markMaterial(const std::uint32_t index) {
        changes |= Change::MaterialValues;
        materials.add(index);
    }
    void markLight(const std::uint32_t index) {
        changes |= Change::Lights;
        lights.add(index);
    }

    bool has(const Change change) const { return (changes & change) != Change::None; }
    bool isEmpty() const { return changes == Change::None; }
    Change getChanges() const { return changes; }
    const Range& getInstanceRange() const { return instances; }
    const Range& getMeshRange() const { return meshes; }
    const Range& getMaterialRange() const { return materials; }
    const Range& getLightRange() const { return lights; }

private:
    Change changes = Change::None;
    Range instances;
    Range meshes;
    Range materials;
    Range lights;
};

}
