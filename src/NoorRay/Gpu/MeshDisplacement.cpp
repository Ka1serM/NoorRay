#include "NoorRay/Gpu/MeshDisplacement.h"

#include <algorithm>
#include <unordered_set>

#include "NoorRHI/Commands/CommandList.h"
#include "NoorRHI/Device/Device.h"
#include "NoorRay/Core/Hash.h"
#include "NoorRay/Core/Log.h"
#include "NoorRay/Gpu/BufferUpload.h"
#include "NoorRay/Gpu/GraphBakes.h"
#include "NoorRay/Gpu/MaterialTable.h"
#include "NoorRay/Gpu/MeshBuffers.h"
#include "NoorRay/Gpu/TextureTable.h"
#include "NoorRay/Materials/Graph.h"
#include "NoorRay/Materials/GraphShader.h"
#include "NoorRay/Materials/Program.h"
#include "NoorRay/Scene/MeshTopology.h"
#include "NoorRayShadersVariants.h"

namespace NoorRay::Gpu {

namespace {

template <class T>
std::uint64_t getKey(const Core::Handle<T> handle) {
    return std::uint64_t{handle.generation} << 32 | handle.index;
}

//! The first material of the mesh with a displacement program; Cycles displaces each face with its own material, which one program per mesh simplifies.
const Scene::Material* findDisplacing(const Scene::Scene& scene, const Scene::Mesh& mesh, Scene::MaterialHandle& handle) {
    for (const Scene::MeshSubset& subset : mesh.subsets) {
        const Scene::Material* material = scene.find(subset.material);
        if (material && material->program && material->program->getDisplacement()) {
            handle = subset.material;
            return material;
        }
    }
    return nullptr;
}

//! The transform of the first instance of each mesh.
std::unordered_map<std::uint64_t, Core::Transform> findTransforms(const Scene::Scene& scene) {
    std::unordered_map<std::uint64_t, Core::Transform> transforms;
    for (const Scene::Instance& instance : scene.getInstances().getValues())
        transforms.try_emplace(getKey(instance.mesh), instance.transform);
    return transforms;
}

bool readsPosition(const Materials::Graph& graph) {
    return graph.reads(Materials::GraphOutput::Emission, Materials::GraphInput::Position)
           || graph.reads(Materials::GraphOutput::Emission, Materials::GraphInput::ObjectLocation);
}

//! The displacement graphs of a mesh and their values, 0 without a graph; a turn of the object changes a world space offset in object space.
std::uint64_t hashGraphs(const Scene::Scene& scene, const Scene::Mesh& mesh, Core::Transform transform) {
    std::uint64_t hash = Core::hashSeed;
    bool found = false;
    bool placed = false;
    for (const Scene::MeshSubset& subset : mesh.subsets) {
        const Scene::Material* material = scene.find(subset.material);
        if (!material || !material->displacementGraph)
            continue;
        found = true;
        placed = placed || readsPosition(*material->displacementGraph);
        hash = Core::combineHash(hash, getKey(subset.material));
        hash = Core::combineHash(hash, hashGraphValues(*material->displacementGraph, material->displacementParameters, material->displacementTextures));
    }
    if (!found)
        return 0;
    if (!placed)
        for (std::size_t row = 0; row < 3; ++row)
            transform.rows[row][3] = 0.0f;
    return Core::hashValue(transform, hash);
}

std::span<const float> asFloats(const std::span<const Core::float3> values) {
    static_assert(sizeof(Core::float3) == 3 * sizeof(float));
    return {reinterpret_cast<const float*>(values.data()), values.size() * 3};
}

template <class T>
NoorRHI::Ptr<T> getPointer(const NoorRHI::Buffer<T>& buffer) {
    return buffer ? buffer.getPtr() : NoorRHI::Ptr<T>{};
}

NoorRHI::DispatchSize getDispatch(const std::uint32_t count) {
    const std::uint32_t blocks = std::max<std::uint32_t>((count + Shaders::displacementGroupSize - 1) / Shaders::displacementGroupSize, 1);
    const std::uint32_t rows = (blocks + Shaders::displacementRowGroups - 1) / Shaders::displacementRowGroups;
    return {std::min(blocks, Shaders::displacementRowGroups) * Shaders::displacementGroupSize, rows, 1};
}

std::uint32_t getStream(const std::vector<std::uint32_t>& stream, const Scene::MeshDomain domain) {
    if (stream.empty())
        return Shaders::displacementStreamNone;
    return domain == Scene::MeshDomain::Vertex ? Shaders::displacementStreamVertex : Shaders::displacementStreamCorner;
}

}

std::expected<void, Core::Error> MeshDisplacement::openPipelines(NoorRHI::Device& device) {
    const std::array<const NoorRHI::Shader*, 4> shaders{Shaders::displaceVertices[0], Shaders::displaceFaceNormals[0], Shaders::displaceVertexNormals[0],
                                                        Shaders::displaceScatter[0]};
    for (std::size_t index = 0; index < shaders.size(); ++index)
        if (auto result = Core::check(pipelines[index].open(device, *shaders[index])); !result)
            return result;
    opened = true;
    return {};
}

std::expected<void, Core::Error> MeshDisplacement::prepare(NoorRHI::Device& device, const Scene::Mesh& mesh, const Materials::Program* program,
                                                           Entry& entry) {
    const Scene::MeshTopology& topology = *mesh.topology;
    const std::span<const std::string> names = program ? std::span<const std::string>(program->getAttributes()) : std::span<const std::string>();
    const std::uint32_t vertexCount = mesh.getVertexCount();
    const std::vector<Core::float3> faceNormals = topology.computeFaceNormals(mesh.positions);
    const std::vector<Core::float3> restNormals = topology.computeVertexNormals(mesh.positions, faceNormals);

    //! A scalar attribute fills its whole value, like a scalar register; a missing attribute reads zero, as in Blender.
    std::vector<Core::float4> attributes(std::max<std::size_t>(names.size() * vertexCount, 1));
    for (std::size_t index = 0; index < names.size(); ++index) {
        const auto found = std::ranges::find(mesh.attributes, names[index], &Scene::MeshAttribute::name);
        if (found == mesh.attributes.end()) {
            Core::Log::warning("mesh '{}' has no attribute '{}' for its displacement", mesh.name, names[index]);
            continue;
        }
        for (std::uint32_t vertex = 0; vertex < vertexCount; ++vertex) {
            const float* value = found->values.data() + std::size_t{vertex} * found->components;
            float components[4]{value[0], value[0], value[0], value[0]};
            if (found->components > 1)
                for (std::uint8_t component = 0; component < 4; ++component)
                    components[component] = component < found->components ? value[component] : 0.0f;
            attributes[index * vertexCount + vertex] = {components[0], components[1], components[2], components[3]};
        }
    }

    const std::span<const std::uint32_t> cornerVertices = topology.getCornerVertices();
    const std::span<const std::uint32_t> faceOffsets = topology.getFaceOffsets();
    std::vector<std::uint32_t> cornerOffsets(vertexCount + 1, 0);
    for (const std::uint32_t vertex : cornerVertices)
        ++cornerOffsets[vertex + 1];
    for (std::uint32_t vertex = 0; vertex < vertexCount; ++vertex)
        cornerOffsets[vertex + 1] += cornerOffsets[vertex];
    std::vector<NoorRHI::uint2> corners(std::max<std::size_t>(cornerVertices.size(), 1));
    std::vector<std::uint32_t> filled(cornerOffsets.begin(), cornerOffsets.end() - 1);
    for (std::uint32_t face = 0; face + 1 < faceOffsets.size(); ++face)
        for (std::uint32_t corner = faceOffsets[face]; corner < faceOffsets[face + 1]; ++corner)
            corners[filled[cornerVertices[corner]]++] = {face, corner};

    const std::vector<float> vertexSpace(std::size_t{vertexCount} * 3, 0.0f);
    const std::vector<float> faceSpace(topology.getFaceCount() * 3, 0.0f);
    if (program)
        if (auto result = uploadValues<NoorRHI::uint2>(device, entry.instructions, program->getInstructions(), false); !result)
            return result;
    if (auto result = uploadValues<NoorRHI::float4>(device, entry.attributes, attributes, false); !result)
        return result;
    if (auto result = uploadValues<float>(device, entry.restPositions, asFloats(mesh.positions), false); !result)
        return result;
    if (auto result = uploadValues<float>(device, entry.restNormals, asFloats(restNormals), false); !result)
        return result;
    if (auto result = uploadValues<float>(device, entry.faceNormals, faceSpace, false); !result)
        return result;
    if (auto result = uploadValues<float>(device, entry.vertexNormals, vertexSpace, false); !result)
        return result;
    if (auto result = uploadValues<std::uint32_t>(device, entry.cornerVertices, cornerVertices, false); !result)
        return result;
    if (auto result = uploadValues<std::uint32_t>(device, entry.faceOffsets, faceOffsets, false); !result)
        return result;
    if (auto result = uploadValues<std::uint32_t>(device, entry.cornerOffsets, cornerOffsets, false); !result)
        return result;
    if (auto result = uploadValues<NoorRHI::uint2>(device, entry.corners, corners, false); !result)
        return result;
    if (!mesh.tangents.empty())
        if (auto result = uploadValues<std::uint32_t>(device, entry.restTangents, mesh.tangents, false); !result)
            return result;
    if (!entry.record)
        if (auto result = Core::check(entry.record.open(device, 1)); !result)
            return result;

    Shaders::DisplacementRecord& values = entry.values;
    values.instructions = getPointer(entry.instructions);
    values.attributes = entry.attributes.getPtr();
    values.restPositions = entry.restPositions.getPtr();
    values.restNormals = entry.restNormals.getPtr();
    values.faceNormals = entry.faceNormals.getPtr();
    values.vertexNormals = entry.vertexNormals.getPtr();
    values.cornerVertices = entry.cornerVertices.getPtr();
    values.faceOffsets = entry.faceOffsets.getPtr();
    values.cornerOffsets = entry.cornerOffsets.getPtr();
    values.corners = entry.corners.getPtr();
    values.restTangents = mesh.tangents.empty() ? NoorRHI::Ptr<std::uint32_t>{} : entry.restTangents.getPtr();
    const bool bounded = mesh.bounds.min.x <= mesh.bounds.max.x;
    values.boundsMin = bounded ? Core::float4{mesh.bounds.min.x, mesh.bounds.min.y, mesh.bounds.min.z, 0.0f} : Core::float4{};
    values.boundsMax = bounded ? Core::float4{mesh.bounds.max.x, mesh.bounds.max.y, mesh.bounds.max.z, 0.0f} : Core::float4{1.0f, 1.0f, 1.0f, 0.0f};
    values.instructionCount = program ? static_cast<std::uint32_t>(program->getInstructions().size()) : 0;
    values.registerCount = program ? static_cast<std::uint32_t>(program->getRegisters().size()) : 1;
    values.displacement = program ? *program->getDisplacement() : 0;
    values.normalStream = getStream(mesh.normals, mesh.normalDomain);
    values.tangentStream = getStream(mesh.tangents, mesh.tangentDomain);
    values.vertexCount = vertexCount;
    values.faceCount = static_cast<std::uint32_t>(topology.getFaceCount());
    values.cornerCount = static_cast<std::uint32_t>(mesh.indices.size());
    entry.registers.clear();
    return {};
}

std::expected<const NoorRHI::ComputePipeline*, Core::Error> MeshDisplacement::getKernel(NoorRHI::Device& device, const Materials::Graph& graph) {
    std::expected<Materials::GraphCode, Core::Error> code = Materials::writeGraphCode(graph);
    if (!code)
        return std::unexpected(std::move(code.error()));
    if (const auto found = kernels.find(code->key); found != kernels.end())
        return &found->second->pipeline;
    auto kernel = std::make_unique<Kernel>();
    if (auto spliced = Materials::spliceGraphShader(*Shaders::displaceGraph[0], *code, kernel->code, kernel->shader); !spliced)
        return std::unexpected(std::move(spliced.error()));
    if (auto result = Core::check(kernel->pipeline.open(device, kernel->shader)); !result)
        return std::unexpected(std::move(result.error()));
    return &kernels.emplace(code->key, std::move(kernel)).first->second->pipeline;
}

std::expected<void, Core::Error> MeshDisplacement::prepareGraphs(NoorRHI::Device& device, const Scene::Scene& scene, const Scene::Mesh& mesh, Entry& entry) {
    entry.graphs.clear();
    std::vector<bool> taken(mesh.getVertexCount());
    const std::vector<Core::float3> vertexNormals = mesh.topology->computeVertexNormals(mesh.positions, mesh.topology->computeFaceNormals(mesh.positions));
    for (const Scene::MeshSubset& subset : mesh.subsets) {
        const Scene::Material* material = scene.find(subset.material);
        if (!material || !material->displacementGraph)
            continue;
        std::vector<NoorRHI::uint2> points;
        std::vector<Core::float3> normals;
        for (std::uint32_t corner = subset.firstIndex; corner < subset.firstIndex + subset.indexCount; ++corner) {
            const std::uint32_t vertex = mesh.indices[corner];
            if (taken[vertex])
                continue;
            taken[vertex] = true;
            const std::uint32_t first = corner - corner % 3;
            const Core::float3 p0 = mesh.positions[mesh.indices[first]];
            const Core::float3 cross = Core::cross(mesh.positions[mesh.indices[first + 1]] - p0, mesh.positions[mesh.indices[first + 2]] - p0);
            const float size = Core::length(cross);
            const Core::float3 flat = size > 0.0f ? cross / size : vertexNormals[vertex];
            points.push_back({vertex, corner});
            const bool smooth = !mesh.normals.empty();
            const bool corners = mesh.normalDomain == Scene::MeshDomain::Corner;
            normals.push_back(!smooth ? flat : corners ? Core::unpackOctahedral(mesh.normals[corner]) : vertexNormals[vertex]);
            normals.push_back(flat);
        }
        if (points.empty())
            continue;
        GraphPass pass;
        pass.material = subset.material;
        pass.pointCount = static_cast<std::uint32_t>(points.size());
        if (auto result = uploadValues<NoorRHI::uint2>(device, pass.points, points, false); !result)
            return result;
        if (auto result = uploadValues<float>(device, pass.normals, asFloats(normals), false); !result)
            return result;
        entry.graphs.push_back(std::move(pass));
    }
    return {};
}

std::expected<void, Core::Error> MeshDisplacement::updateGraphs(NoorRHI::Device& device, const Scene::Scene& scene, const Scene::Mesh& mesh,
                                                                const Core::Transform& transform, const TextureTable& textures, MeshGpu& gpu, Entry& entry) {
    const Core::Transform inverse = transform.inverse();
    for (GraphPass& pass : entry.graphs) {
        const Scene::Material& material = *scene.find(pass.material);
        std::expected<const NoorRHI::ComputePipeline*, Core::Error> pipeline = getKernel(device, *material.displacementGraph);
        if (!pipeline)
            return std::unexpected(std::move(pipeline.error()));
        pass.pipeline = *pipeline;
        std::vector<Shaders::MaterialTextureRecord> textureRecords;
        for (const Scene::MaterialTexture& texture : material.displacementTextures)
            textureRecords.push_back(makeTextureRecord(texture, textures));
        if (auto result = uploadValues<NoorRHI::float4>(device, pass.parameters, material.displacementParameters, false); !result)
            return result;
        if (auto result = uploadValues<Shaders::MaterialTextureRecord>(device, pass.textures, textureRecords, false); !result)
            return result;
        const auto uv = [&](const std::uint32_t set) {
            return (mesh.uvAliases >> set & 1u) != 0 ? getPointer(gpu.uvs[0]) : mesh.uvs[set].empty() ? NoorRHI::Ptr<std::uint32_t>{} : gpu.uvs[set].getPtr();
        };
        const auto bit = [](const Scene::MeshDomain domain, const std::uint32_t value) { return domain == Scene::MeshDomain::Corner ? value : 0u; };
        Shaders::DisplacementGraphRecord values{};
        values.mesh = entry.record.getPtr();
        values.parameters = pass.parameters.getPtr();
        values.textures = pass.textures.getPtr();
        values.points = pass.points.getPtr();
        values.normals = pass.normals.getPtr();
        values.uv0 = uv(0);
        values.uv1 = uv(1);
        values.uv2 = uv(2);
        values.uv3 = uv(3);
        values.colors = mesh.colors.empty() ? NoorRHI::Ptr<std::uint32_t>{} : gpu.colors.getPtr();
        values.pointCount = pass.pointCount;
        values.cornerStreams = bit(mesh.uvDomain, Shaders::meshCornerUvs) | bit(mesh.tangentDomain, Shaders::meshCornerTangents)
                               | bit(mesh.colorDomain, Shaders::meshCornerColors);
        for (std::size_t row = 0; row < 3; ++row) {
            values.objectToWorld[row] = {transform.rows[row][0], transform.rows[row][1], transform.rows[row][2], transform.rows[row][3]};
            values.worldToObject[row] = {inverse.rows[row][0], inverse.rows[row][1], inverse.rows[row][2], inverse.rows[row][3]};
        }
        if (auto result = uploadValues<Shaders::DisplacementGraphRecord>(device, pass.record, std::span(&values, 1), false); !result)
            return result;
    }
    return {};
}

std::expected<bool, Core::Error> MeshDisplacement::update(NoorRHI::Device& device, const Scene::Scene& scene, const Scene::SceneChange& change,
                                                          const bool everything, const TextureTable& textures, MeshBuffers& meshes) {
    const Core::SlotMap<Scene::Mesh>& sceneMeshes = scene.getMeshes();
    const bool allMeshes = everything || change.has(Scene::Change::MeshTopology);
    const bool materials = everything || change.has(Scene::Change::MaterialSet | Scene::Change::MaterialValues | Scene::Change::Textures);
    const bool placed = change.has(Scene::Change::InstanceTransforms | Scene::Change::InstanceSet);
    if (!allMeshes && !materials && !change.has(Scene::Change::MeshPositions | Scene::Change::MeshReplaced) && (!placed || entries.empty()))
        return false;
    std::unordered_map<std::uint64_t, Core::Transform> transforms;
    if (std::ranges::any_of(scene.getMaterials().getValues(), [](const Scene::Material& material) { return material.displacementGraph != nullptr; }))
        transforms = findTransforms(scene);
    const Scene::SceneChange::IndexSet& moved = change.getMeshes();
    const Scene::SceneChange::IndexSet& replaced = change.getReplacedMeshes();
    std::unordered_set<std::uint64_t> live;
    std::vector<std::pair<Entry*, MeshGpu*>> pending;
    for (std::uint32_t index = 0; index < sceneMeshes.size() && index < meshes.getCount(); ++index) {
        const Scene::Mesh& mesh = sceneMeshes.getValues()[index];
        const std::uint64_t key = getKey(sceneMeshes.keyAt(index));
        Scene::MaterialHandle handle;
        const Scene::Material* material = findDisplacing(scene, mesh, handle);
        const auto transform = transforms.find(key);
        const Core::Transform place = transform == transforms.end() ? Core::Transform{} : transform->second;
        const std::uint64_t graphKey = hashGraphs(scene, mesh, place);
        MeshGpu& gpu = meshes.getMesh(index);
        if (!material && graphKey == 0) {
            gpu.restPositions = {};
            gpu.restNormals = {};
            continue;
        }
        if (!mesh.topology) {
            if (!std::ranges::contains(warned, key)) {
                Core::Log::warning("mesh '{}' has a displacing material but no source topology; it is not displaced", mesh.name);
                warned.push_back(key);
            }
            continue;
        }
        live.insert(key);
        const auto [found, created] = entries.try_emplace(key);
        Entry& entry = found->second;
        const Materials::Program* program = material ? material->program.get() : nullptr;
        const bool reset = created || allMeshes || moved.contains(index) || replaced.contains(index) || entry.material != handle || entry.program != program;
        if (reset) {
            entry.material = handle;
            entry.program = program;
            if (auto result = prepare(device, mesh, program, entry); !result)
                return std::unexpected(std::move(result.error()));
        }
        //! Without a program, one zero register leaves each vertex at its rest position.
        const std::vector<Core::float4> registers = material ? material->programRegisters : std::vector<Core::float4>(1);
        if (!reset && entry.graphKey == graphKey && std::ranges::equal(entry.registers, registers))
            continue;
        if (reset || entry.graphKey != graphKey)
            if (auto result = prepareGraphs(device, scene, mesh, entry); !result)
                return std::unexpected(std::move(result.error()));
        entry.graphKey = graphKey;
        entry.registers = registers;
        if (auto result = uploadValues<NoorRHI::float4>(device, entry.registerValues, std::span<const Core::float4>(entry.registers), false); !result)
            return std::unexpected(std::move(result.error()));
        gpu.restPositions = entry.restPositions.getPtr();
        gpu.restNormals = entry.restNormals.getPtr();
        entry.values.registers = entry.registerValues.getPtr();
        entry.values.positions = gpu.positions.getPtr().as<float>();
        entry.values.indices = gpu.indices.getPtr();
        entry.values.normals = getPointer(gpu.normals);
        entry.values.tangents = getPointer(gpu.tangents);
        if (mesh.tangents.empty())
            entry.values.restTangents = {};
        if (auto result = Core::check(entry.record.upload(std::span(&entry.values, 1))); !result)
            return std::unexpected(std::move(result.error()));
        if (auto result = updateGraphs(device, scene, mesh, place, textures, gpu, entry); !result)
            return std::unexpected(std::move(result.error()));
        pending.emplace_back(&entry, &gpu);
    }
    std::erase_if(entries, [&](const auto& item) { return !live.contains(item.first); });
    if (pending.empty())
        return false;
    if (!opened)
        if (auto result = openPipelines(device); !result)
            return std::unexpected(std::move(result.error()));
    NoorRHI::CommandList list(device);
    list.pushLabel("MeshDisplacement");
    for (const auto& [entry, gpu] : pending) {
        const Shaders::DisplacementArguments arguments{entry->record.getPtr()};
        const Shaders::DisplacementRecord& values = entry->values;
        list.dispatch(pipelines[0], getDispatch(values.vertexCount), arguments);
        for (const GraphPass& pass : entry->graphs)
            list.dispatch(*pass.pipeline, getDispatch(pass.pointCount), Shaders::DisplacementGraphArguments{pass.record.getPtr()});
        list.dispatch(pipelines[1], getDispatch(values.faceCount), arguments);
        list.dispatch(pipelines[2], getDispatch(values.vertexCount), arguments);
        if (values.normalStream == Shaders::displacementStreamCorner || values.tangentStream == Shaders::displacementStreamCorner)
            list.dispatch(pipelines[3], getDispatch(values.cornerCount), arguments);
        gpu->blasRefit = true;
    }
    list.popLabel();
    //! The BLAS refit that follows reads the positions on the device, so the displacement finishes first.
    auto token = Core::check(device.submit(std::move(list)));
    if (!token)
        return std::unexpected(std::move(token.error()));
    if (auto waited = Core::check(device.wait(*token)); !waited)
        return std::unexpected(std::move(waited.error()));
    return true;
}

}
