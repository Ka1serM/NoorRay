#include "NoorRay/Format/AnimationRecorder.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <format>
#include <limits>
#include <ranges>
#include <type_traits>

#include "NoorRay/Core/Hash.h"
#include "NoorRay/Materials/Graph.h"
#include "NoorRay/Materials/Program.h"
#include "NoorRay/Scene/MeshTopology.h"

namespace NoorRay::Format {

namespace {

constexpr double keyTolerance = 1e-6;

template <class Value>
struct Property {
    std::string_view name;
    Value* values = nullptr;
    std::size_t components = 1;
};

template <class Object>
auto propertiesOf(Object& object) {
    using Type = std::remove_const_t<Object>;
    using Entry = Property<std::conditional_t<std::is_const_v<Object>, const float, float>>;
    if constexpr (std::is_same_v<Type, Scene::Instance>)
        return std::array{Entry{"transform", &object.transform.rows[0][0], 12}};
    else if constexpr (std::is_same_v<Type, Scene::Light>)
        return std::array{Entry{"transform", &object.transform.rows[0][0], 12},
                          Entry{"color", &object.color.x, 3},
                          Entry{"power", &object.power},
                          Entry{"radius", &object.radius},
                          Entry{"width", &object.width},
                          Entry{"height", &object.height},
                          Entry{"spot_angle", &object.spotAngle},
                          Entry{"spot_blend", &object.spotBlend},
                          Entry{"angle", &object.angle},
                          Entry{"spread", &object.spread}};
    else if constexpr (std::is_same_v<Type, Scene::Camera>)
        return std::array{Entry{"transform", &object.transform.rows[0][0], 12},
                          Entry{"vertical_fov", &object.verticalFov},
                          Entry{"orthographic_height", &object.orthographicHeight},
                          Entry{"focus_distance", &object.focusDistance},
                          Entry{"f_stop", &object.fStop},
                          Entry{"sensor_width", &object.sensor.width},
                          Entry{"sensor_height", &object.sensor.height},
                          Entry{"sensor_offset_x", &object.sensor.offset.x},
                          Entry{"sensor_offset_y", &object.sensor.offset.y},
                          Entry{"aperture_rotation", &object.apertureRotation},
                          Entry{"aperture_ratio", &object.apertureRatio}};
    else if constexpr (std::is_same_v<Type, Scene::Material>)
        return std::array{Entry{"base_color", &object.baseColor.x, 3},
                          Entry{"alpha", &object.alpha},
                          Entry{"alpha_cutoff", &object.alphaCutoff},
                          Entry{"metallic", &object.metallic},
                          Entry{"roughness", &object.roughness},
                          Entry{"normal_scale", &object.normalScale},
                          Entry{"ior", &object.ior},
                          Entry{"specular_ior_level", &object.specularIorLevel},
                          Entry{"specular_tint", &object.specularTint.x, 3},
                          Entry{"transmission", &object.transmission},
                          Entry{"anisotropic", &object.anisotropic},
                          Entry{"anisotropic_rotation", &object.anisotropicRotation},
                          Entry{"coat", &object.coat},
                          Entry{"coat_roughness", &object.coatRoughness},
                          Entry{"coat_normal_scale", &object.coatNormalScale},
                          Entry{"sheen_color", &object.sheenColor.x, 3},
                          Entry{"sheen_roughness", &object.sheenRoughness},
                          Entry{"emission_color", &object.emissionColor.x, 3},
                          Entry{"emission_strength", &object.emissionStrength},
                          Entry{"thin_film_thickness", &object.thinFilmThickness},
                          Entry{"thin_film_ior", &object.thinFilmIor},
                          Entry{"dispersion", &object.dispersion}};
    else if constexpr (std::is_same_v<Type, Scene::Environment>)
        return std::array{Entry{"color", &object.color.x, 3}, Entry{"strength", &object.strength}, Entry{"rotation", &object.rotation}};
    else
        return std::array{Entry{"transform", &object.transform.rows[0][0], 12}, Entry{"absorption", &object.absorption.x, 3},
                          Entry{"scattering", &object.scattering.x, 3},         Entry{"emission", &object.emission.x, 3},
                          Entry{"emission_rate", &object.emissionRate.x, 3},    Entry{"anisotropy", &object.anisotropy},
                          Entry{"density_scale", &object.densityScale}};
}

template <class Object>
void copyProperties(const Object& from, Object& to) {
    const auto sources = propertiesOf(from);
    const auto targets = propertiesOf(to);
    for (std::size_t index = 0; index < sources.size(); ++index)
        std::ranges::copy_n(sources[index].values, sources[index].components, targets[index].values);
}

//! Cycles' random id comes from the name and the persistent ids, so it names an instance across frames.
std::uint64_t identityOf(const Scene::Instance& instance) {
    return Core::floatToBits(instance.random);
}

template <class Object>
std::uint64_t identityOf(const Object& object) {
    return Core::hashString(object.name);
}

bool isSame(const Core::uint3 a, const Core::uint3 b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

bool isSameGraph(const std::shared_ptr<const Materials::Graph>& a, const std::shared_ptr<const Materials::Graph>& b) {
    return a == b || (a && b && a->hasSameProgram(*b));
}

bool isSameProgram(const Materials::Program* a, const Materials::Program* b) {
    if (a == b)
        return true;
    if (!a || !b)
        return false;
    const auto sameParameter = [](const Materials::ProgramParameter& x, const Materials::ProgramParameter& y) {
        return x.reg == y.reg && x.components == y.components;
    };
    return std::ranges::equal(a->getInstructions(), b->getInstructions(), [](const Core::uint2 x, const Core::uint2 y) { return x.x == y.x && x.y == y.y; })
           && std::ranges::equal(a->getParameters(), b->getParameters(), sameParameter) && std::ranges::equal(a->getAttributes(), b->getAttributes())
           && a->getDisplacement() == b->getDisplacement();
}

//! Equal values from the first slot on that has no channel.
bool hasSameSlots(const std::vector<Core::float4>& a, const std::vector<Core::float4>& b, const std::size_t channels) {
    return a.size() == b.size() && std::ranges::equal(a | std::views::drop(channels), b | std::views::drop(channels));
}

std::size_t channelParameters(const Scene::Material& base, const Scene::Material& value) {
    if (!base.program || !value.program)
        return 0;
    const std::span<const Materials::ProgramParameter> a = base.program->getParameters();
    const std::span<const Materials::ProgramParameter> b = value.program->getParameters();
    std::size_t count = 0;
    while (count < std::min(a.size(), b.size()) && a[count].components == b[count].components)
        ++count;
    return count;
}

bool hasSameRegisters(const Scene::Material& a, const Scene::Material& b, const std::size_t channels) {
    if (!a.program)
        return true;
    std::vector<Core::float4> registers = a.programRegisters;
    if (registers.size() != b.programRegisters.size())
        return false;
    for (const Materials::ProgramParameter& parameter : a.program->getParameters().first(channels))
        registers[parameter.reg] = b.programRegisters[parameter.reg];
    return registers == b.programRegisters;
}

//! Whether the channels, which start from base, express every difference. The node tree only describes the first frame.
bool channelsExpress(const Scene::Material& base, const Scene::Material& a, const Scene::Material& b) {
    return a.name == b.name && a.useVertexColor == b.useVertexColor && a.alphaMode == b.alphaMode && a.roughnessIsGlossiness == b.roughnessIsGlossiness
           && a.emissionSampling == b.emissionSampling && a.textures == b.textures && isSameProgram(a.program.get(), b.program.get())
           && hasSameRegisters(a, b, channelParameters(base, b)) && isSameGraph(a.graph, b.graph)
           && hasSameSlots(a.graphParameters, b.graphParameters, base.graphParameters.size()) && a.graphTextures == b.graphTextures
           && isSameGraph(a.displacementGraph, b.displacementGraph)
           && hasSameSlots(a.displacementParameters, b.displacementParameters, base.displacementParameters.size())
           && a.displacementTextures == b.displacementTextures && a.bumpsRestNormal == b.bumpsRestNormal && a.bumpsSurface == b.bumpsSurface;
}

bool channelsExpress(const Scene::Volume& base, const Scene::Volume& a, const Scene::Volume& b) {
    return a.name == b.name && a.kind == b.kind && a.bounds.min == b.bounds.min && a.bounds.max == b.bounds.max && a.draineAlpha == b.draineAlpha
           && isSame(a.resolution, b.resolution) && isSame(a.maskResolution, b.maskResolution) && a.lightSet == b.lightSet
           && isSameGraph(a.densityGraph, b.densityGraph) && hasSameSlots(a.densityParameters, b.densityParameters, base.densityParameters.size())
           && a.densityTextures == b.densityTextures && a.density == b.density && a.emissionGrid == b.emissionGrid && a.mask == b.mask;
}

bool channelsExpress(const Scene::Instance&, const Scene::Instance& a, const Scene::Instance& b) {
    constexpr auto others = ~std::to_underlying(Scene::InstanceFlags::Hidden);
    return a.mesh == b.mesh && (std::to_underlying(a.flags) & others) == (std::to_underlying(b.flags) & others) && a.material == b.material
           && a.lightSet == b.lightSet && a.lightMask == b.lightMask && a.color == b.color && a.passIndex == b.passIndex && a.random == b.random
           && a.attributes == b.attributes;
}

bool channelsExpress(const Scene::Camera&, const Scene::Camera& a, const Scene::Camera& b) {
    Scene::Camera same = a;
    copyProperties(b, same);
    same.previousTransform = b.previousTransform;
    return same == b;
}

template <class Object>
    requires(std::is_same_v<Object, Scene::Light> || std::is_same_v<Object, Scene::Environment>)
bool channelsExpress(const Object& base, const Object& a, const Object& b) {
    if (!isSameGraph(a.graph, b.graph) || !hasSameSlots(a.graphParameters, b.graphParameters, base.graphParameters.size()))
        return false;
    Object same = a;
    copyProperties(b, same);
    same.graph = b.graph;
    same.graphParameters = b.graphParameters;
    same.nodeTree = b.nodeTree;
    return same == b;
}

Scene::Light absentLight(Scene::Light light) {
    light.power = 0.0f;
    light.visibleToCamera = false;
    return light;
}

Scene::Volume absentVolume(Scene::Volume volume) {
    volume.absorption = {};
    volume.scattering = {};
    volume.emission = {};
    volume.emissionRate = {};
    volume.densityScale = 0.0f;
    volume.emissionGrid.clear();
    return volume;
}

bool isSameTopology(const Scene::MeshTopology* a, const Scene::MeshTopology* b) {
    if (a == b)
        return true;
    return a && b && a->getNormalSource() == b->getNormalSource() && std::ranges::equal(a->getCornerVertices(), b->getCornerVertices())
           && std::ranges::equal(a->getFaceOffsets(), b->getFaceOffsets());
}

bool isSameAttribute(const Scene::MeshAttribute& a, const Scene::MeshAttribute& b) {
    return a.name == b.name && a.components == b.components && a.values == b.values;
}

//! Whether new positions, normals and tangents are all that changes full into live, whose subsets have the materials of the output scene.
bool needsOnlyVertices(const Scene::Mesh& full, const Scene::Mesh& live, const std::span<const Scene::MeshSubset> subsets) {
    return full.positions.size() == live.positions.size() && full.normals.size() == live.normals.size() && full.tangents.size() == live.tangents.size()
           && full.normalDomain == live.normalDomain && full.tangentDomain == live.tangentDomain && full.uvDomain == live.uvDomain
           && full.colorDomain == live.colorDomain && full.uvAliases == live.uvAliases && std::ranges::equal(full.subsets, subsets)
           && full.indices == live.indices && full.uvs == live.uvs && full.colors == live.colors
           && std::ranges::equal(full.attributes, live.attributes, isSameAttribute) && isSameTopology(full.topology.get(), live.topology.get());
}

struct VertexStreams {
    std::span<const Core::float3> positions;
    std::span<const std::uint32_t> normals;
    std::span<const std::uint32_t> tangents;
};

//! The streams that a vertex frame of the mesh holds; its topology rebuilds the rest.
VertexStreams storedStreams(const Scene::Mesh& mesh) {
    if (!mesh.topology)
        return {mesh.positions, mesh.normals, mesh.tangents};
    if (mesh.topology->getNormalSource() == Scene::NormalSource::Stored)
        return {mesh.positions, mesh.normals, {}};
    return {mesh.positions, {}, {}};
}

VertexStreams lastStreams(const MeshFrame& frame) {
    if (frame.mesh)
        return storedStreams(*frame.mesh);
    return {frame.positions, frame.normals, frame.tangents};
}

bool isSame(const VertexStreams& a, const VertexStreams& b) {
    return std::ranges::equal(a.positions, b.positions) && std::ranges::equal(a.normals, b.normals) && std::ranges::equal(a.tangents, b.tangents);
}

//! Drops the keys that the line between their neighbours passes: from each kept key, the range of slopes through the keys after it narrows until one misses.
class KeyReducer {
public:
    KeyReducer(const std::span<const float> frames, const std::span<const float> values, const std::size_t components)
        : frames(frames), values(values), components(components), low(components), high(components), tolerances(components) {
        std::array<float, 2> largest{};
        for (std::size_t index = 0; index < values.size(); ++index)
            largest[group(index % components)] = std::max(largest[group(index % components)], std::abs(values[index]));
        for (std::size_t component = 0; component < components; ++component)
            tolerances[component] = keyTolerance * largest[group(component)];
    }

    std::vector<std::uint32_t> reduce() {
        std::vector<std::uint32_t> kept{0};
        const auto count = static_cast<std::uint32_t>(frames.size());
        open();
        for (std::uint32_t key = 1; key < count; ++key) {
            if (!reaches(kept.back(), key)) {
                kept.push_back(key - 1);
                open();
            }
            narrow(kept.back(), key);
        }
        if (kept.back() + 1 < count)
            kept.push_back(count - 1);
        return kept;
    }

private:
    //! The translation column of a transform has another scale than its axes.
    std::size_t group(const std::size_t component) const { return components == 12 && component % 4 == 3 ? 1 : 0; }
    void open() {
        std::ranges::fill(low, -std::numeric_limits<double>::infinity());
        std::ranges::fill(high, std::numeric_limits<double>::infinity());
    }
    double slope(const std::uint32_t anchor, const std::uint32_t key, const std::size_t component, const double offset) const {
        const double start = values[anchor * components + component];
        return (values[key * components + component] + offset - start) / (frames[key] - frames[anchor]);
    }
    bool reaches(const std::uint32_t anchor, const std::uint32_t key) const {
        for (std::size_t component = 0; component < components; ++component) {
            const double value = slope(anchor, key, component, 0.0);
            if (!(value >= low[component] && value <= high[component]))
                return false;
        }
        return true;
    }
    void narrow(const std::uint32_t anchor, const std::uint32_t key) {
        for (std::size_t component = 0; component < components; ++component) {
            low[component] = std::max(low[component], slope(anchor, key, component, -tolerances[component]));
            high[component] = std::min(high[component], slope(anchor, key, component, tolerances[component]));
        }
    }

    std::span<const float> frames;
    std::span<const float> values;
    std::size_t components;
    std::vector<double> low;
    std::vector<double> high;
    std::vector<double> tolerances;
};

std::span<const float> floatsOf(const Core::float4& value) {
    return {&value.x, 4};
}

template <class Object>
const Object& lastValue(const std::vector<ObjectValue>& keys, const Object& base) {
    return keys.empty() ? base : std::get<Object>(keys.back());
}

}

template <class Object>
std::uint32_t AnimationRecorder::Table<Object>::recordOf(const Handle live) const {
    if (!live || live.index >= slots.size() || slots[live.index] == Handle::invalidIndex || records[slots[live.index]].live != live)
        return Handle::invalidIndex;
    return slots[live.index];
}

template <class Object>
void AnimationRecorder::Table<Object>::attach(const std::uint32_t record, const Handle live) {
    if (live.index >= slots.size())
        slots.resize(live.index + 1, Handle::invalidIndex);
    slots[live.index] = record;
    records[record].live = live;
}

template <class Object>
template <class Keep>
void AnimationRecorder::Table<Object>::release(Keep&& keep) {
    for (std::uint32_t index = 0; index < records.size(); ++index) {
        Record<Object>& record = records[index];
        if (record.live && !keep(index, record.live))
            record.live = {};
    }
}

template <class Object>
std::vector<std::uint32_t> AnimationRecorder::Table<Object>::attachNew(const Core::SlotMap<Object>& live, const Core::SlotMap<Object>& output) {
    std::vector<std::uint32_t> fresh;
    for (std::uint32_t dense = 0; dense < live.size(); ++dense)
        if (recordOf(live.keyAt(dense)) == Handle::invalidIndex)
            fresh.push_back(dense);
    if (fresh.empty())
        return fresh;
    //! The lists are in descending order, so that the first free record is at the back.
    std::unordered_map<std::uint64_t, std::vector<std::uint32_t>> free;
    for (std::uint32_t record = static_cast<std::uint32_t>(records.size()); record-- > 0;)
        if (!records[record].live)
            free[identityOf(output.getValues()[record])].push_back(record);
    std::vector<std::uint32_t> attached;
    attached.reserve(fresh.size());
    for (const std::uint32_t dense : fresh) {
        const auto found = free.find(identityOf(live.getValues()[dense]));
        std::uint32_t record = static_cast<std::uint32_t>(records.size());
        if (found != free.end() && !found->second.empty()) {
            record = found->second.back();
            found->second.pop_back();
        } else {
            records.emplace_back();
        }
        attach(record, live.keyAt(dense));
        attached.push_back(record);
    }
    return attached;
}

void AnimationRecorder::record(const Scene::Scene& live, const int frame) {
    if (!started)
        animation.firstFrame = frame;
    animation.lastFrame = frame;
    trees.clear();
    attachTextures(live);
    recordMaterials(live, frame);
    recordInstances(live, frame);
    recordLights(live, frame);
    recordVolumes(live, frame);
    recordCameras(live, frame);
    recordEnvironment(live, frame);
    started = true;
}

//! The handle of a texture never gets other texels, so only new handles are compared: a texture that is read again stays one texture.
void AnimationRecorder::attachTextures(const Scene::Scene& live) {
    textures.release([&](std::uint32_t, const Scene::TextureHandle handle) { return live.find(handle) != nullptr; });
    for (std::uint32_t dense = 0; dense < live.getTextures().size(); ++dense) {
        const Scene::TextureHandle handle = live.getTextures().keyAt(dense);
        if (textures.find(handle))
            continue;
        const Scene::Texture& texture = live.getTextures().getValues()[dense];
        std::uint32_t match = 0;
        while (match < textures.records.size()) {
            const Scene::Texture& known = output.getTextures().getValues()[match];
            if (!textures.records[match].live && known.name == texture.name && known.format == texture.format && known.width == texture.width
                && known.height == texture.height && known.data == texture.data)
                break;
            ++match;
        }
        if (match == textures.records.size()) {
            textures.records.emplace_back();
            output.addTexture(texture);
        }
        textures.attach(match, handle);
    }
}

void AnimationRecorder::recordMaterials(const Scene::Scene& live, const int frame) {
    materials.release([&](std::uint32_t, const Scene::MaterialHandle handle) { return live.find(handle) != nullptr; });
    materials.attachNew(live.getMaterials(), output.getMaterials());
    for (std::uint32_t index = 0; index < materials.records.size(); ++index) {
        Record<Scene::Material>& record = materials.records[index];
        const Scene::Material* source = live.find(record.live);
        if (!source)
            continue;
        Scene::Material value = remapped(*source);
        if (index == output.getMaterials().size()) {
            value.nodeTree = remapped(value.nodeTree);
            output.addMaterial(std::move(value));
            continue;
        }
        const Scene::Material& base = output.getMaterials().getValues()[index];
        sampleProperties(record, base, value, frame);
        sampleSlots(record.channels, "graph", base.graphParameters, value.graphParameters, frame);
        sampleSlots(record.channels, "displacement", base.displacementParameters, value.displacementParameters, frame);
        sampleParameters(record, base, value, frame);
        const Scene::Material& last = lastValue(record.keyValues, base);
        if (channelsExpress(base, last, value))
            continue;
        if (isSameGraph(last.graph, value.graph))
            value.graph = last.graph;
        if (isSameGraph(last.displacementGraph, value.displacementGraph))
            value.displacementGraph = last.displacementGraph;
        value.nodeTree = remapped(value.nodeTree);
        addKey(record, base, std::move(value), frame);
    }
}

void AnimationRecorder::recordMeshes(const Scene::Scene& live, const std::span<const std::uint32_t> newInstances, const int frame) {
    meshes.release([&](std::uint32_t, const Scene::MeshHandle handle) { return live.find(handle) != nullptr; });
    //! An instance that comes back with another mesh continues the mesh it had, when nothing else shows that one.
    for (const std::uint32_t index : newInstances) {
        if (index >= output.getInstances().size())
            continue;
        const Record<Scene::Instance>& record = instances.records[index];
        const Scene::MeshHandle mesh = live.find(record.live)->mesh;
        const Scene::MeshHandle last = output.getInstances().getValues()[index].mesh;
        if (last && live.find(mesh) && !meshes.records[last.index].live && !meshes.find(mesh))
            meshes.attach(last.index, mesh);
    }
    const std::vector<std::uint32_t> attached = meshes.attachNew(live.getMeshes(), output.getMeshes());
    const Scene::SceneChange& change = live.getChange();
    //! A removal moves dense indices, and a removed material clears subsets, so the exact dirty meshes do not hold then.
    const bool all = attached.size() > 0 || change.has(Scene::Change::MeshTopology | Scene::Change::MaterialSet);
    for (std::uint32_t index = 0; index < meshes.records.size(); ++index) {
        const Scene::MeshHandle handle = meshes.records[index].live;
        const Scene::Mesh* source = live.find(handle);
        if (!source)
            continue;
        if (index == output.getMeshes().size()) {
            output.addMesh(remapped(*source));
            meshStates.push_back({.firstFrame = frame});
            continue;
        }
        const std::uint32_t dense = live.getMeshes().denseIndex(handle);
        if (all || change.getMeshes().contains(dense) || change.getReplacedMeshes().contains(dense))
            recordMesh(index, *source, frame);
    }
}

void AnimationRecorder::recordMesh(const std::uint32_t record, const Scene::Mesh& live, const int frame) {
    MeshState& state = meshStates[record];
    const Scene::Mesh& base = output.getMeshes().getValues()[record];
    MeshSequence* sequence = state.sequence ? &animation.meshSequences[*state.sequence] : nullptr;
    const Scene::Mesh& full = sequence ? *sequence->frames[state.fullFrame].mesh : base;
    const bool vertices = needsOnlyVertices(full, live, remapped(live.subsets));
    const VertexStreams streams = storedStreams(live);
    if (vertices && isSame(streams, sequence ? lastStreams(sequence->frames.back()) : storedStreams(base)))
        return;
    if (!sequence) {
        state.sequence = animation.meshSequences.size();
        sequence = &animation.meshSequences.emplace_back(MeshSequence{.mesh = record});
        sequence->frames.push_back({.frame = state.firstFrame, .mesh = base});
    }
    if (vertices) {
        sequence->frames.push_back({.frame = frame,
                                    .positions = {streams.positions.begin(), streams.positions.end()},
                                    .normals = {streams.normals.begin(), streams.normals.end()},
                                    .tangents = {streams.tangents.begin(), streams.tangents.end()}});
        return;
    }
    state.fullFrame = sequence->frames.size();
    sequence->frames.push_back({.frame = frame, .mesh = remapped(live)});
}

void AnimationRecorder::recordInstances(const Scene::Scene& live, const int frame) {
    instances.release([&](std::uint32_t, const Scene::InstanceHandle handle) { return live.find(handle) != nullptr; });
    const std::vector<std::uint32_t> attached = instances.attachNew(live.getInstances(), output.getInstances());
    recordMeshes(live, attached, frame);
    splitInstances(live, attached);
    constexpr float shown = 0.0f;
    constexpr float hidden = 1.0f;
    for (std::uint32_t index = 0; index < instances.records.size(); ++index) {
        Record<Scene::Instance>& record = instances.records[index];
        const Scene::Instance* source = live.find(record.live);
        if (!source && !record.present)
            continue;
        const bool isNew = index == output.getInstances().size();
        if (isNew) {
            Scene::Instance added = remapped(*source);
            if (started)
                added.flags = added.flags | Scene::InstanceFlags::Hidden;
            output.addInstance(std::move(added));
        }
        const Scene::Instance& base = output.getInstances().getValues()[index];
        const float& baseHidden = Scene::hasFlag(base.flags, Scene::InstanceFlags::Hidden) ? hidden : shown;
        const bool isHidden = !source || Scene::hasFlag(source->flags, Scene::InstanceFlags::Hidden);
        sample(record.channels, "hidden", noSlot, std::span(&baseHidden, 1), std::span(isHidden ? &hidden : &shown, 1), true, frame);
        record.present = source != nullptr;
        if (!source || isNew)
            continue;
        Scene::Instance value = remapped(*source);
        value.id = base.id;
        sampleProperties(record, base, value, frame);
        if (!channelsExpress(base, lastValue(record.keyValues, base), value))
            addKey(record, base, std::move(value), frame);
    }
}

//! An output instance keeps its mesh, so a live one that shows another mesh moves to a free record with that mesh or to a new one.
void AnimationRecorder::splitInstances(const Scene::Scene& live, const std::span<const std::uint32_t> attached) {
    const std::span<const Scene::Instance> known = output.getInstances().getValues();
    const auto keyOf = [](const Scene::Instance& instance, const Scene::MeshHandle mesh) { return Core::combineHash(identityOf(instance), mesh.index); };
    std::vector<std::uint32_t> moved;
    for (const std::uint32_t index : attached)
        if (index < known.size() && known[index].mesh != meshes.find(live.find(instances.records[index].live)->mesh))
            moved.push_back(index);
    if (moved.empty())
        return;
    std::vector<Scene::InstanceHandle> handles;
    for (const std::uint32_t index : moved)
        handles.push_back(std::exchange(instances.records[index].live, {}));
    std::unordered_map<std::uint64_t, std::vector<std::uint32_t>> free;
    for (std::uint32_t index = static_cast<std::uint32_t>(known.size()); index-- > 0;)
        if (!instances.records[index].live)
            free[keyOf(known[index], known[index].mesh)].push_back(index);
    for (const Scene::InstanceHandle handle : handles) {
        const Scene::Instance& instance = *live.find(handle);
        const Scene::MeshHandle mesh = meshes.find(instance.mesh);
        const auto found = free.find(keyOf(instance, mesh));
        if (found != free.end() && !found->second.empty() && known[found->second.back()].mesh == mesh) {
            instances.attach(found->second.back(), handle);
            found->second.pop_back();
            continue;
        }
        instances.records.emplace_back();
        instances.attach(static_cast<std::uint32_t>(instances.records.size() - 1), handle);
    }
}

void AnimationRecorder::recordLights(const Scene::Scene& live, const int frame) {
    lights.release([&](std::uint32_t, const Scene::LightHandle handle) { return live.find(handle) != nullptr; });
    lights.attachNew(live.getLights(), output.getLights());
    for (std::uint32_t index = 0; index < lights.records.size(); ++index) {
        Record<Scene::Light>& record = lights.records[index];
        const Scene::Light* source = live.find(record.live);
        if (!source && !record.present)
            continue;
        if (index == output.getLights().size()) {
            Scene::Light added = remapped(*source);
            added.nodeTree = remapped(added.nodeTree);
            output.addLight(started ? absentLight(std::move(added)) : std::move(added));
        }
        const Scene::Light& base = output.getLights().getValues()[index];
        const Scene::Light& last = lastValue(record.keyValues, base);
        record.present = source != nullptr;
        Scene::Light value = source ? remapped(*source) : absentLight(last);
        if (source) {
            sampleProperties(record, base, value, frame);
            sampleSlots(record.channels, "graph", base.graphParameters, value.graphParameters, frame);
        } else {
            sample(record.channels, "power", noSlot, std::span(&base.power, 1), std::span(&value.power, 1), false, frame);
        }
        if (channelsExpress(base, last, value))
            continue;
        if (isSameGraph(last.graph, value.graph))
            value.graph = last.graph;
        if (source)
            value.nodeTree = remapped(value.nodeTree);
        addKey(record, base, std::move(value), frame);
    }
}

void AnimationRecorder::recordVolumes(const Scene::Scene& live, const int frame) {
    if (started && !live.getChange().has(Scene::Change::Volumes))
        return;
    volumes.release([&](std::uint32_t, const Scene::VolumeHandle handle) { return live.find(handle) != nullptr; });
    volumes.attachNew(live.getVolumes(), output.getVolumes());
    for (std::uint32_t index = 0; index < volumes.records.size(); ++index) {
        Record<Scene::Volume>& record = volumes.records[index];
        const Scene::Volume* source = live.find(record.live);
        if (!source && !record.present)
            continue;
        if (index == output.getVolumes().size())
            output.addVolume(started ? absentVolume(remapped(*source)) : remapped(*source));
        const Scene::Volume& base = output.getVolumes().getValues()[index];
        const Scene::Volume& last = lastValue(record.keyValues, base);
        record.present = source != nullptr;
        Scene::Volume value = source ? remapped(*source) : absentVolume(last);
        if (source) {
            sampleProperties(record, base, value, frame);
            sampleSlots(record.channels, "density", base.densityParameters, value.densityParameters, frame);
        } else {
            const auto bases = propertiesOf(base);
            const auto values = propertiesOf(std::as_const(value));
            for (std::size_t property = 0; property < values.size(); ++property)
                if (values[property].name != "transform" && values[property].name != "anisotropy")
                    sample(record.channels, values[property].name, noSlot, std::span(bases[property].values, bases[property].components),
                           std::span(values[property].values, values[property].components), false, frame);
        }
        if (channelsExpress(base, last, value))
            continue;
        if (isSameGraph(last.densityGraph, value.densityGraph))
            value.densityGraph = last.densityGraph;
        addKey(record, base, std::move(value), frame);
    }
}

//! The live scene has one camera object for every Blender camera, so a new name is a cut to another camera.
void AnimationRecorder::recordCameras(const Scene::Scene& live, const int frame) {
    cameras.release([&](const std::uint32_t index, const Scene::CameraHandle handle) {
        const Scene::Camera* camera = live.find(handle);
        return camera && camera->name == output.getCameras().getValues()[index].name;
    });
    cameras.attachNew(live.getCameras(), output.getCameras());
    for (std::uint32_t index = 0; index < cameras.records.size(); ++index) {
        Record<Scene::Camera>& record = cameras.records[index];
        const Scene::Camera* source = live.find(record.live);
        if (!source)
            continue;
        if (index == output.getCameras().size()) {
            output.addCamera(*source);
            continue;
        }
        const Scene::Camera& base = output.getCameras().getValues()[index];
        sampleProperties(record, base, *source, frame);
        if (!channelsExpress(base, lastValue(record.keyValues, base), *source))
            addKey(record, base, *source, frame);
    }
    const Scene::CameraHandle active = cameras.find(live.getActiveCamera());
    if (!active)
        return;
    const auto value = static_cast<float>(active.index);
    if (!started) {
        firstActiveCamera = value;
        output.setActiveCamera(active);
    }
    sample(sceneChannels, "active_camera", noSlot, std::span(&firstActiveCamera, 1), std::span(&value, 1), true, frame);
}

void AnimationRecorder::recordEnvironment(const Scene::Scene& live, const int frame) {
    if (started && !live.getChange().has(Scene::Change::Environment))
        return;
    Scene::Environment value = remapped(live.getEnvironment());
    if (!started) {
        value.nodeTree = remapped(value.nodeTree);
        output.setEnvironment(value);
        return;
    }
    const Scene::Environment& base = output.getEnvironment();
    sampleProperties(environment, base, value, frame);
    sampleSlots(environment.channels, "graph", base.graphParameters, value.graphParameters, frame);
    const Scene::Environment& last = lastValue(environment.keyValues, base);
    if (channelsExpress(base, last, value))
        return;
    if (isSameGraph(last.graph, value.graph))
        value.graph = last.graph;
    value.nodeTree = remapped(value.nodeTree);
    addKey(environment, base, std::move(value), frame);
}

void AnimationRecorder::sample(std::vector<Channel>& channels, const std::string_view name, const std::uint32_t slot, const std::span<const float> base,
                               const std::span<const float> value, const bool constant, const int frame) const {
    const auto found = std::ranges::find_if(channels, [&](const Channel& channel) { return channel.slot == slot && channel.name == name; });
    const std::size_t size = value.size();
    const float* last = found == channels.end() ? base.data() : found->values.data() + found->values.size() - size;
    if (std::memcmp(last, value.data(), size * sizeof(float)) == 0)
        return;
    Channel& channel = found == channels.end() ? channels.emplace_back(Channel{.name = name,
                                                                               .slot = slot,
                                                                               .components = static_cast<std::uint8_t>(size),
                                                                               .constant = constant,
                                                                               .frames = {static_cast<float>(animation.firstFrame)},
                                                                               .values = {base.begin(), base.end()}})
                                               : *found;
    const auto previous = static_cast<float>(frame - 1);
    if (!constant && channel.frames.back() < previous) {
        channel.frames.push_back(previous);
        const std::size_t first = channel.values.size() - size;
        for (std::size_t component = 0; component < size; ++component)
            channel.values.push_back(channel.values[first + component]);
    }
    channel.frames.push_back(static_cast<float>(frame));
    channel.values.insert(channel.values.end(), value.begin(), value.end());
}

template <class Object>
void AnimationRecorder::sampleProperties(Record<Object>& record, const Object& base, const Object& value, const int frame) const {
    const auto bases = propertiesOf(base);
    const auto values = propertiesOf(value);
    for (std::size_t index = 0; index < values.size(); ++index)
        sample(record.channels, values[index].name, noSlot, std::span(bases[index].values, bases[index].components),
               std::span(values[index].values, values[index].components), false, frame);
}

void AnimationRecorder::sampleSlots(std::vector<Channel>& channels, const std::string_view name, const std::span<const Core::float4> base,
                                    const std::span<const Core::float4> value, const int frame) const {
    const std::size_t count = std::min(base.size(), value.size());
    for (std::uint32_t slot = 0; slot < count; ++slot)
        sample(channels, name, slot, floatsOf(base[slot]), floatsOf(value[slot]), false, frame);
}

void AnimationRecorder::sampleParameters(Record<Scene::Material>& record, const Scene::Material& base, const Scene::Material& value, const int frame) const {
    const std::size_t count = channelParameters(base, value);
    for (std::uint32_t index = 0; index < count; ++index) {
        const Materials::ProgramParameter& from = base.program->getParameters()[index];
        const Materials::ProgramParameter& to = value.program->getParameters()[index];
        sample(record.channels, "parameter", index, floatsOf(base.programRegisters[from.reg]).first(from.components),
               floatsOf(value.programRegisters[to.reg]).first(to.components), false, frame);
    }
}

//! The first key is the value of the first frame, so that the frames before the change have a key too.
template <class Object>
void AnimationRecorder::addKey(Record<Object>& record, const Object& base, Object value, const int frame) const {
    if (record.keyFrames.empty() && frame > animation.firstFrame) {
        record.keyFrames.push_back(animation.firstFrame);
        record.keyValues.emplace_back(base);
    }
    record.keyFrames.push_back(frame);
    record.keyValues.emplace_back(std::move(value));
}

void AnimationRecorder::remap(std::vector<Scene::MaterialTexture>& list) const {
    for (Scene::MaterialTexture& texture : list)
        texture.texture = textures.find(texture.texture);
}

Scene::Material AnimationRecorder::remapped(Scene::Material material) const {
    for (std::optional<Scene::MaterialTexture>& texture : material.textures)
        if (texture)
            texture->texture = textures.find(texture->texture);
    remap(material.graphTextures);
    remap(material.displacementTextures);
    return material;
}

Scene::Light AnimationRecorder::remapped(Scene::Light light) const {
    remap(light.graphTextures);
    return light;
}

Scene::Volume AnimationRecorder::remapped(Scene::Volume volume) const {
    remap(volume.densityTextures);
    return volume;
}

Scene::Environment AnimationRecorder::remapped(Scene::Environment environment) const {
    environment.texture = textures.find(environment.texture);
    remap(environment.graphTextures);
    return environment;
}

Scene::Instance AnimationRecorder::remapped(Scene::Instance instance) const {
    instance.previousTransform = instance.transform;
    instance.mesh = meshes.find(instance.mesh);
    instance.material = materials.find(instance.material);
    return instance;
}

std::vector<Scene::MeshSubset> AnimationRecorder::remapped(const std::span<const Scene::MeshSubset> subsets) const {
    std::vector<Scene::MeshSubset> result(subsets.begin(), subsets.end());
    for (Scene::MeshSubset& subset : result)
        subset.material = materials.find(subset.material);
    return result;
}

Scene::Mesh AnimationRecorder::remapped(Scene::Mesh mesh) const {
    mesh.subsets = remapped(mesh.subsets);
    return mesh;
}

std::shared_ptr<const Materials::NodeTree> AnimationRecorder::remapped(const std::shared_ptr<const Materials::NodeTree>& tree) {
    if (!tree)
        return nullptr;
    if (const auto known = trees.find(tree.get()); known != trees.end())
        return known->second;
    auto result = std::make_shared<Materials::NodeTree>(*tree);
    const auto remapProperties = [&](std::vector<Materials::NodeProperty>& properties) {
        for (Materials::NodeProperty& property : properties)
            property.texture = textures.find(property.texture);
    };
    remapProperties(result->properties);
    for (Materials::Node& node : result->nodes)
        remapProperties(node.properties);
    for (std::shared_ptr<const Materials::NodeTree>& group : result->groups)
        group = remapped(group);
    trees[tree.get()] = result;
    return result;
}

void AnimationRecorder::writeTrack(const AnimationTarget target, const std::uint32_t index, std::vector<Channel>& channels) {
    if (channels.empty())
        return;
    AnimationTrack track{.target = target, .index = index};
    for (Channel& channel : channels) {
        KeyChannel keys{.property = channel.slot == noSlot ? std::string(channel.name) : std::format("{}:{}", channel.name, channel.slot),
                        .components = channel.components};
        if (channel.constant) {
            keys.frames = std::move(channel.frames);
            keys.values = std::move(channel.values);
        } else {
            for (const std::uint32_t key : KeyReducer(channel.frames, channel.values, channel.components).reduce()) {
                keys.frames.push_back(channel.frames[key]);
                const auto first = channel.values.begin() + static_cast<std::ptrdiff_t>(std::size_t{key} * channel.components);
                keys.values.insert(keys.values.end(), first, first + channel.components);
            }
        }
        track.channels.push_back(std::move(keys));
    }
    animation.tracks.push_back(std::move(track));
}

template <class Object>
void AnimationRecorder::write(const AnimationTarget target, const std::span<Record<Object>> records) {
    for (std::uint32_t index = 0; index < records.size(); ++index) {
        Record<Object>& record = records[index];
        writeTrack(target, index, record.channels);
        if (!record.keyFrames.empty())
            animation.objectKeys.push_back({.target = target, .index = index, .frames = std::move(record.keyFrames), .values = std::move(record.keyValues)});
    }
}

RecordedAnimation AnimationRecorder::take(const float framesPerSecond) {
    animation.framesPerSecond = framesPerSecond;
    writeTrack(AnimationTarget::Scene, 0, sceneChannels);
    write(AnimationTarget::Instance, std::span(instances.records));
    write(AnimationTarget::Light, std::span(lights.records));
    write(AnimationTarget::Camera, std::span(cameras.records));
    write(AnimationTarget::Material, std::span(materials.records));
    write(AnimationTarget::Environment, std::span(&environment, 1));
    write(AnimationTarget::Volume, std::span(volumes.records));
    RecordedAnimation result{.scene = std::move(output), .animation = std::move(animation)};
    *this = {};
    return result;
}

}
