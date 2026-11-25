#include "NoorRay/Format/SceneDecoder.h"

#include <algorithm>
#include <format>
#include <limits>

namespace NoorRay::Format {

namespace {

//! The largest degree that keeps the coefficient count far from overflow; Gaussian files stop at degree 3.
constexpr std::uint32_t maxShDegree = 7;

std::uint64_t countOf(const Core::uint3 resolution) {
    if (resolution.x == 0 || resolution.y == 0 || resolution.z == 0)
        return 0;
    const std::uint64_t layer = std::uint64_t{resolution.x} * resolution.y;
    if (layer > std::numeric_limits<std::uint64_t>::max() / resolution.z)
        return std::numeric_limits<std::uint64_t>::max();
    return layer * resolution.z;
}

}

void SceneDecoder::transferSize(std::size_t& size) {
    const std::uint64_t stored = reader.read<std::uint64_t>();
    if (stored > std::numeric_limits<std::size_t>::max())
        refuse("a size past the address space");
    size = static_cast<std::size_t>(stored);
}

void SceneDecoder::transferTopology(Scene::Mesh& mesh) {
    if (!reader.read<bool>())
        return;
    Scene::NormalSource source{};
    std::vector<std::uint32_t> cornerVertices;
    std::vector<std::uint32_t> faceOffsets;
    (*this)(source, cornerVertices, faceOffsets);
    if (!reader.isValid())
        return;
    auto topology = std::make_shared<Scene::MeshTopology>();
    if (std::expected<void, Core::Error> opened = topology->open(source, std::move(cornerVertices), std::move(faceOffsets), mesh); !opened) {
        refuse(opened.error().message);
        return;
    }
    mesh.topology = std::move(topology);
}

void SceneDecoder::validate(const Scene::Texture& texture) {
    if (texture.format == NoorRHI::ImageFormat::Undefined)
        refuse(std::format("the texture '{}' without a format", texture.name));
    for (const Scene::TextureLevel& level : texture.levels)
        if (level.offset > texture.data.size() || level.size > texture.data.size() - level.offset)
            refuse(std::format("the texture '{}' with a level past its data", texture.name));
}

void SceneDecoder::validate(const Scene::Material& material) {
    if (material.program && material.programRegisters.size() != material.program->getRegisters().size())
        refuse(std::format("the material '{}' with {} registers for a program that has {}", material.name, material.programRegisters.size(),
                           material.program->getRegisters().size()));
    validateGraph(material.name, material.graph, material.graphParameters.size(), material.graphTextures.size());
    validateGraph(material.name, material.displacementGraph, material.displacementParameters.size(), material.displacementTextures.size());
}

void SceneDecoder::validate(const Scene::Mesh& mesh) {
    const std::size_t vertexCount = mesh.positions.size();
    if (!mesh.hasValidStreams() || mesh.indices.size() % 3 != 0)
        refuse(std::format("the mesh '{}' with streams that do not match their domains", mesh.name));
    if (std::ranges::any_of(mesh.indices, [&](const std::uint32_t vertex) { return vertex >= vertexCount; }))
        refuse(std::format("the mesh '{}' with an index past its vertices", mesh.name));
    for (const Scene::MeshSubset& subset : mesh.subsets)
        if (std::uint64_t{subset.firstIndex} + subset.indexCount > mesh.indices.size())
            refuse(std::format("the mesh '{}' with a subset past its indices", mesh.name));
    for (const Scene::MeshAttribute& attribute : mesh.attributes)
        if (attribute.components == 0 || attribute.components > 4 || attribute.values.size() % attribute.components != 0
            || attribute.values.size() / attribute.components < vertexCount)
            refuse(std::format("the attribute '{}' of mesh '{}' without a value for each vertex", attribute.name, mesh.name));
}

void SceneDecoder::validate(const Scene::Light& light) {
    if (light.falloffPower > 2)
        refuse(std::format("the light '{}' with an unknown falloff", light.name));
    validateGraph(light.name, light.graph, light.graphParameters.size(), light.graphTextures.size());
}

void SceneDecoder::validate(const Scene::GaussianCloud& cloud) {
    const std::size_t count = cloud.positions.size();
    if (cloud.shDegree > maxShDegree || cloud.scales.size() != count || cloud.rotations.size() != count || cloud.opacities.size() != count
        || cloud.sh.size() != count * Scene::getShCoefficientCount(cloud.shDegree) * 3)
        refuse(std::format("the Gaussian cloud '{}' with arrays of different lengths", cloud.name));
}

void SceneDecoder::validate(const Scene::Volume& volume) {
    const std::uint64_t voxels = countOf(volume.resolution);
    if (volume.kind == Scene::VolumeKind::Grid && (voxels == 0 || (!volume.densityGraph && volume.density.size() != voxels)))
        refuse(std::format("the grid volume '{}' with {} densities for {} voxels and no density graph", volume.name, volume.density.size(), voxels));
    if ((!volume.emissionGrid.empty() && volume.emissionGrid.size() != voxels) || volume.mask.size() != countOf(volume.maskResolution))
        refuse(std::format("the volume '{}' with an emission or a mask of the wrong size", volume.name));
    validateGraph(volume.name, volume.densityGraph, volume.densityParameters.size(), volume.densityTextures.size());
}

void SceneDecoder::validate(const Scene::Environment& environment) {
    validateGraph("environment", environment.graph, environment.graphParameters.size(), environment.graphTextures.size());
}

void SceneDecoder::validate(const Materials::NodeTree& tree) {
    const auto isIndex = [](const std::uint32_t value, const std::size_t count) { return value == Materials::noNode || value < count; };
    for (const Materials::Node& node : tree.nodes)
        if (!isIndex(node.parent, tree.nodes.size()) || !isIndex(node.group, tree.groups.size()))
            refuse(std::format("the node tree '{}' with a frame or group that does not exist", tree.name));
    for (const Materials::NodeLink& link : tree.links)
        if (link.fromNode >= tree.nodes.size() || link.toNode >= tree.nodes.size() || link.fromSocket >= tree.nodes[link.fromNode].outputs.size()
            || link.toSocket >= tree.nodes[link.toNode].inputs.size())
            refuse(std::format("the node tree '{}' with a link to a socket that does not exist", tree.name));
}

void SceneDecoder::validateGraph(const std::string_view owner, const std::shared_ptr<const Materials::Graph>& graph, const std::size_t values,
                                 const std::size_t textures) {
    if (graph && (values != graph->getParameterSlots() || textures != graph->getTextureCount()))
        refuse(std::format("the graph of '{}' with {} values and {} textures, but it needs {} and {}", owner, values, textures, graph->getParameterSlots(),
                           graph->getTextureCount()));
}

void SceneDecoder::refuse(const std::string_view content) {
    reader.invalidate();
    if (!problem)
        problem = Core::Error{Core::ErrorCode::InvalidFile, std::format("the scene file has {}", content)};
}

}
