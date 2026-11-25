#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "NoorRay/Scene/Scene.h"

namespace NoorRay::Format {

//! Every index is a dense index into the arrays of the written scene; Scene has the one index 0.
enum class AnimationTarget : std::uint8_t { Scene, Instance, Light, Camera, Material, Environment, Volume };

//! The keys of one property; frames is sorted and values holds components floats per frame. The property names and their targets are in Animation.cpp.
struct KeyChannel {
    std::string property;
    std::uint8_t components = 1;
    std::vector<float> frames;
    std::vector<float> values;
};

struct AnimationTrack {
    AnimationTarget target = AnimationTarget::Scene;
    std::uint32_t index = 0;
    std::vector<KeyChannel> channels;
};

using ObjectValue = std::variant<Scene::Instance, Scene::Light, Scene::Camera, Scene::Material, Scene::Environment, Scene::Volume>;

//! Whole-object values for changes that no channel expresses; a frame shows the last key at or before it, and channels of the object apply after it.
struct ObjectKeys {
    AnimationTarget target = AnimationTarget::Scene;
    std::uint32_t index = 0;
    std::vector<int> frames;
    std::vector<ObjectValue> values;
};

//! A full mesh when topology, streams or subsets change; else vertex streams for the last full mesh, with empty normals and tangents when its topology rebuilds them.
struct MeshFrame {
    int frame = 0;
    std::optional<Scene::Mesh> mesh;
    std::vector<Core::float3> positions;
    std::vector<std::uint32_t> normals;
    std::vector<std::uint32_t> tangents;
};

//! The first frame holds a full mesh. Frames are sorted, and a mesh keeps its last frame until the next one.
struct MeshSequence {
    std::uint32_t mesh = 0;
    std::vector<MeshFrame> frames;
};

struct AnimationData {
    int firstFrame = 0;
    int lastFrame = 0;
    float framesPerSecond = 24.0f;
    std::vector<AnimationTrack> tracks;
    std::vector<ObjectKeys> objectKeys;
    std::vector<MeshSequence> meshSequences;

    bool isEmpty() const { return tracks.empty() && objectKeys.empty() && meshSequences.empty(); }
};

}
