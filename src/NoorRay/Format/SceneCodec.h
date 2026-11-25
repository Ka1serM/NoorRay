#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

#include "NoorRay/Format/AnimationData.h"
#include "NoorRay/Materials/Graph.h"
#include "NoorRay/Materials/NodeTree.h"
#include "NoorRay/Materials/Program.h"
#include "NoorRay/Scene/MeshTopology.h"
#include "NoorRay/Scene/Scene.h"

namespace NoorRay::Format {

inline constexpr std::array<char, 4> fileMagic{'N', 'R', 'S', 'C'};
inline constexpr std::uint32_t fileVersion = 1;

//! The start of a scene file. The scene follows it, then the frame area, the shared objects and the animation directory.
struct FileHeader {
    std::array<char, 4> magic = fileMagic;
    std::uint32_t version = fileVersion;
    std::uint64_t size = 0;
    std::uint64_t sharedOffset = 0;
    //! 0 for a still scene.
    std::uint64_t animationOffset = 0;
};
static_assert(sizeof(FileHeader) == 32);

//! The bytes of a key frame in the file, at a 16-byte offset.
struct FramePlace {
    std::uint64_t offset = 0;
    std::uint64_t size = 0;
};
static_assert(sizeof(FramePlace) == 16);

struct ObjectKeyDirectory {
    AnimationTarget target = AnimationTarget::Scene;
    std::uint32_t index = 0;
    std::vector<int> frames;
    std::vector<FramePlace> places;
};

struct MeshSequenceDirectory {
    std::uint32_t mesh = 0;
    std::vector<int> frames;
    std::vector<int> fullFrames;
    std::vector<FramePlace> places;
};

struct AnimationDirectory {
    int firstFrame = 0;
    int lastFrame = 0;
    float framesPerSecond = 24.0f;
    std::vector<AnimationTrack> tracks;
    std::vector<ObjectKeyDirectory> objectKeys;
    std::vector<MeshSequenceDirectory> meshSequences;
};

struct VertexFrame {
    std::vector<Core::float3> positions;
    std::vector<std::uint32_t> normals;
    std::vector<std::uint32_t> tangents;
};

//! Records without padding, handles or pointers, which the file stores as their bytes.
template <class T>
inline constexpr bool isPlainRecord = false;
template <>
inline constexpr bool isPlainRecord<Core::float2> = true;
template <>
inline constexpr bool isPlainRecord<Core::float3> = true;
template <>
inline constexpr bool isPlainRecord<Core::float4> = true;
template <>
inline constexpr bool isPlainRecord<Core::uint2> = true;
template <>
inline constexpr bool isPlainRecord<Core::uint3> = true;
template <>
inline constexpr bool isPlainRecord<Core::Transform> = true;
template <>
inline constexpr bool isPlainRecord<Core::Bounds> = true;
template <>
inline constexpr bool isPlainRecord<Scene::Sensor> = true;
template <>
inline constexpr bool isPlainRecord<Scene::LensElement> = true;
template <>
inline constexpr bool isPlainRecord<Scene::UvTransform> = true;
template <>
inline constexpr bool isPlainRecord<Materials::GraphInstruction> = true;
template <>
inline constexpr bool isPlainRecord<FramePlace> = true;

static_assert(sizeof(Core::float3) == 12 && sizeof(Core::float4) == 16 && sizeof(Core::Transform) == 48 && sizeof(Core::Bounds) == 24);
static_assert(sizeof(Scene::Sensor) == 16 && sizeof(Scene::LensElement) == 20 && sizeof(Scene::UvTransform) == 24);
static_assert(sizeof(Materials::GraphInstruction) == 12 && sizeof(int) == 4);

template <class T>
concept StoredAsBytes = std::is_arithmetic_v<T> || std::is_enum_v<T> || isPlainRecord<T>;

//! The reader refuses a value of an enumeration past its last one.
template <class T>
inline constexpr T lastValue{};
template <>
inline constexpr auto lastValue<NoorRHI::ImageFormat> = NoorRHI::ImageFormat::Bc7Srgb;
template <>
inline constexpr auto lastValue<Scene::TextureWrap> = Scene::TextureWrap::Mirror;
template <>
inline constexpr auto lastValue<Scene::TextureFilter> = Scene::TextureFilter::Cubic;
template <>
inline constexpr auto lastValue<Scene::AlphaMode> = Scene::AlphaMode::Blend;
template <>
inline constexpr auto lastValue<Scene::EmissionSampling> = Scene::EmissionSampling::FrontAndBack;
template <>
inline constexpr auto lastValue<Scene::MeshDomain> = Scene::MeshDomain::Corner;
template <>
inline constexpr auto lastValue<Scene::NormalSource> = Scene::NormalSource::Stored;
template <>
inline constexpr auto lastValue<Scene::InstanceFlags> = static_cast<Scene::InstanceFlags>(UINT32_MAX);
template <>
inline constexpr auto lastValue<Scene::LightType> = Scene::LightType::Directional;
template <>
inline constexpr auto lastValue<Scene::VolumeKind> = Scene::VolumeKind::Grid;
template <>
inline constexpr auto lastValue<Scene::CameraType> = Scene::CameraType::CentralCylindrical;
template <>
inline constexpr auto lastValue<Materials::NodeSocketType> = Materials::NodeSocketType::Custom;
template <>
inline constexpr auto lastValue<Materials::NodePropertyKind> = Materials::NodePropertyKind::Texture;
template <>
inline constexpr auto lastValue<AnimationTarget> = AnimationTarget::Volume;

template <class T, class Object>
concept Of = std::same_as<std::remove_const_t<T>, Object>;

//! SceneEncoder writes and SceneDecoder reads the fields that transfer lists, in this order.
void transfer(auto& archive, Of<Scene::TextureLevel> auto& level) {
    archive.transferSize(level.offset);
    archive.transferSize(level.size);
    archive(level.width, level.height);
}

void transfer(auto& archive, Of<Scene::Texture> auto& texture) {
    archive(texture.name, texture.format, texture.width, texture.height, texture.levels, texture.data);
}

void transfer(auto& archive, Of<Scene::MaterialTexture> auto& texture) {
    archive(texture.texture, texture.wrap, texture.filter, texture.uvSet, texture.channel, texture.transform);
}

void transfer(auto& archive, Of<Scene::Material> auto& material) {
    archive(material.name, material.baseColor, material.useVertexColor, material.alpha, material.alphaMode, material.alphaCutoff, material.metallic,
            material.roughness, material.roughnessIsGlossiness, material.normalScale, material.ior, material.specularIorLevel, material.specularTint,
            material.transmission, material.anisotropic, material.anisotropicRotation, material.coat, material.coatRoughness, material.coatNormalScale,
            material.sheenColor, material.sheenRoughness, material.emissionColor, material.emissionStrength, material.thinFilmThickness, material.thinFilmIor,
            material.dispersion, material.emissionSampling, material.textures);
    archive(material.program, material.programRegisters, material.graph, material.graphParameters, material.graphTextures, material.displacementGraph,
            material.displacementParameters, material.displacementTextures, material.bumpsRestNormal, material.bumpsSurface, material.nodeTree);
}

void transfer(auto& archive, Of<Scene::MeshSubset> auto& subset) {
    archive(subset.firstIndex, subset.indexCount, subset.material);
}

void transfer(auto& archive, Of<Scene::MeshAttribute> auto& attribute) {
    archive(attribute.name, attribute.components, attribute.values);
}

void transfer(auto& archive, Of<Scene::Mesh> auto& mesh) {
    archive(mesh.name, mesh.positions, mesh.normals, mesh.tangents, mesh.uvs, mesh.uvAliases, mesh.colors, mesh.normalDomain, mesh.tangentDomain, mesh.uvDomain,
            mesh.colorDomain, mesh.indices, mesh.subsets, mesh.bounds, mesh.attributes);
    archive.transferTopology(mesh);
}

void transfer(auto& archive, Of<Scene::Instance> auto& instance) {
    archive(instance.transform, instance.previousTransform, instance.mesh, instance.flags, instance.id, instance.material, instance.lightSet,
            instance.lightMask, instance.color, instance.passIndex, instance.random, instance.attributes);
}

void transfer(auto& archive, Of<Scene::Light> auto& light) {
    archive(light.name, light.type, light.transform, light.color, light.power, light.radius, light.width, light.height, light.spotAngle, light.spotBlend,
            light.angle, light.falloffPower, light.falloffSmooth, light.castsShadow, light.maxBounces, light.hidden, light.visibleToCamera, light.spread,
            light.graph, light.graphParameters, light.graphTextures, light.nodeTree, light.lightMask);
}

void transfer(auto& archive, Of<Scene::GaussianCloud> auto& cloud) {
    archive(cloud.name, cloud.transform, cloud.positions, cloud.scales, cloud.rotations, cloud.opacities, cloud.shDegree, cloud.sh);
}

void transfer(auto& archive, Of<Scene::Volume> auto& volume) {
    archive(volume.name, volume.kind, volume.transform, volume.bounds, volume.absorption, volume.scattering, volume.emission, volume.emissionRate,
            volume.anisotropy, volume.draineAlpha, volume.densityScale, volume.resolution, volume.density, volume.emissionGrid, volume.maskResolution,
            volume.mask, volume.densityGraph, volume.densityParameters, volume.densityTextures, volume.lightSet);
}

void transfer(auto& archive, Of<Scene::Camera> auto& camera) {
    archive(camera.name, camera.type, camera.transform, camera.previousTransform, camera.verticalFov, camera.orthographicHeight, camera.sensor,
            camera.focusDistance, camera.fStop, camera.blades, camera.apertureRotation, camera.apertureRatio, camera.near, camera.far, camera.lens.elements,
            camera.panorama, camera.pixelAspect, camera.window);
}

void transfer(auto& archive, Of<Scene::Environment> auto& environment) {
    archive(environment.texture, environment.color, environment.strength, environment.rotation, environment.visibleToCamera, environment.opaque,
            environment.sampled, environment.hidden, environment.filter, environment.graph, environment.graphParameters, environment.graphTextures,
            environment.nodeTree);
}

void transfer(auto& archive, Of<Materials::GraphParameter> auto& parameter) {
    archive(parameter.name, parameter.slot, parameter.width);
}

void transfer(auto& archive, Of<Materials::Graph::Source> auto& graph) {
    archive(graph.instructions, graph.operands, graph.constants, graph.parameters, graph.parameterSlots, graph.textureCount, graph.outputs);
}

void transfer(auto& archive, Of<Materials::ProgramParameter> auto& parameter) {
    archive(parameter.name, parameter.reg, parameter.components);
}

void transfer(auto& archive, Of<Materials::Program::Source> auto& program) {
    archive(program.registers, program.instructions, program.parameters, program.attributes, program.displacement);
}

void transfer(auto& archive, Of<Materials::NodeSocket> auto& socket) {
    archive(socket.identifier, socket.name, socket.type, socket.components, socket.enabled, socket.hidden, socket.value, socket.text);
}

void transfer(auto& archive, Of<Materials::NodeProperty> auto& property) {
    archive(property.name, property.kind, property.number, property.integer, property.text, property.numbers, property.texture);
}

void transfer(auto& archive, Of<Materials::Node> auto& node) {
    archive(node.type, node.name, node.label, node.location, node.width, node.muted, node.hidden, node.parent, node.group, node.inputs, node.outputs,
            node.properties);
}

void transfer(auto& archive, Of<Materials::NodeLink> auto& link) {
    archive(link.fromNode, link.fromSocket, link.toNode, link.toSocket, link.muted);
}

void transfer(auto& archive, Of<Materials::NodeTree> auto& tree) {
    archive(tree.name, tree.nodes, tree.links, tree.properties, tree.groups);
}

void transfer(auto& archive, Of<VertexFrame> auto& frame) {
    archive(frame.positions, frame.normals, frame.tangents);
}

void transfer(auto& archive, Of<KeyChannel> auto& channel) {
    archive(channel.property, channel.components, channel.frames, channel.values);
}

void transfer(auto& archive, Of<AnimationTrack> auto& track) {
    archive(track.target, track.index, track.channels);
}

void transfer(auto& archive, Of<ObjectKeyDirectory> auto& keys) {
    archive(keys.target, keys.index, keys.frames, keys.places);
}

void transfer(auto& archive, Of<MeshSequenceDirectory> auto& sequence) {
    archive(sequence.mesh, sequence.frames, sequence.fullFrames, sequence.places);
}

void transfer(auto& archive, Of<AnimationDirectory> auto& animation) {
    archive(animation.firstFrame, animation.lastFrame, animation.framesPerSecond, animation.tracks, animation.objectKeys, animation.meshSequences);
}

}
