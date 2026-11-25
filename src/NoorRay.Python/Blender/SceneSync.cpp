#include "NoorRay.Python/Blender/SceneSync.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <map>
#include <numbers>
#include <set>
#include <unordered_map>

#include "NoorRay.Python/Blender/BlenderTextures.h"
#include "NoorRay.Python/Blender/CyclesHash.h"
#include "NoorRay.Python/Blender/NodeTreeReader.h"
#include "NoorRay/Materials/Nodes/LightCompiler.h"
#include "NoorRay/Materials/Nodes/NodeAccess.h"
#include "NoorRay/Shaders/Common/GpuRecords.h"

namespace NoorRay::Python {

namespace {

//! The lightHidden bits of the rays that a light or a world does not light, by the names of its flags for diffuse, glossy, transmission and volume scatter.
std::uint32_t hiddenRays(Reader& reader, const Object& owner, const std::array<const char*, 4>& names) {
    std::uint32_t hidden = 0;
    for (std::uint32_t ray = 0; ray < names.size(); ++ray)
        hidden |= reader.flag(owner, names[ray]) ? 0u : 1u << ray;
    return hidden;
}

//! Blender is Z-up and NoorRay is Y-up: the rows (x, z, -y) of the Blender matrix.
Core::Transform toYUp(const Core::Transform& blender) {
    Core::Transform result;
    for (int column = 0; column < 4; ++column) {
        result.rows[0][column] = blender.rows[0][column];
        result.rows[1][column] = blender.rows[2][column];
        result.rows[2][column] = -blender.rows[1][column];
    }
    return result;
}

//! The tree of the default material, whose settings give the fixed material of Scene::Material.
std::shared_ptr<const Materials::NodeTree> defaultTree() {
    auto tree = std::make_shared<Materials::NodeTree>();
    constexpr float color[] = {0.8f, 0.8f, 0.8f, 1.0f};
    tree->properties = {booleanProperty("use_nodes", false), numbersProperty("diffuse_color", color), numberProperty("metallic", 0.0f),
                        numberProperty("roughness", 0.5f)};
    return tree;
}

//! The voxels of a density grid along its longest side.
constexpr std::uint32_t gridResolution = 256;

//! The cells along the longest side of the mask of a volume that is not a box.
constexpr float maskSide = 96.0f;

//! The half size of the volume of a world, in metres.
constexpr float worldVolumeSize = 1e5f;

//! The original of tree and of every group tree it reaches.
void collectTrees(Reader& reader, const Object& tree, std::vector<std::uintptr_t>& trees) {
    if (!tree)
        return;
    const std::uintptr_t original = reader.pointer(reader.get(tree, "original"));
    if (reader.failed() || std::ranges::contains(trees, original))
        return;
    trees.push_back(original);
    reader.forEach(reader.get(tree, "nodes"), [&](const Object& node) {
        if (reader.text(node, "type") == "GROUP")
            collectTrees(reader, reader.find(node, "node_tree"), trees);
    });
}

template <class Key, class Record>
void removeOlder(std::map<Key, Record>& records, Scene::Scene& scene, const std::uint64_t generation) {
    std::erase_if(records, [&](const auto& entry) {
        if (entry.second.generation == generation)
            return false;
        scene.remove(entry.second.handle);
        return true;
    });
}

//! Cycles' texture limit of Simplify, in texels.
std::uint32_t textureLimit(Reader& reader, const Object& depsgraph) {
    const Object scene = reader.get(depsgraph, "scene");
    const Object cycles = reader.find(scene, "cycles");
    if (!cycles || !reader.flag(reader.get(scene, "render"), "use_simplify"))
        return 0;
    const std::string limit = reader.text(cycles, reader.text(depsgraph, "mode") == "RENDER" ? "texture_limit_render" : "texture_limit");
    return static_cast<std::uint32_t>(std::atoi(limit.c_str()));
}

}

struct SceneSync::LightLinking {
    std::vector<Emitter> emitters;
    std::unordered_map<std::uintptr_t, std::size_t> emitterIndices;
    //! The emitters that reach each set, by index into emitters.
    std::vector<std::vector<std::uint32_t>> signatures;
    std::map<std::vector<std::uint32_t>, std::uint32_t> sets;
    std::map<std::pair<std::uintptr_t, bool>, std::uint32_t> receivers;
    bool overflow = false;
};

struct SceneSync::Pass {
    Pass(Scene::Scene& scene, const Object& depsgraph, const bool approximate, TextureTable& textures, Materials::AttributeTables& attributeTables)
        : scene(scene), depsgraph(depsgraph), diagnostics(reader, approximate), trees(reader),
          materials(reader, diagnostics, textures, scene, attributeTables, trees, images), meshes(reader), strands(reader) {}

    Scene::Scene& scene;
    Object depsgraph;
    Reader reader;
    Diagnostics diagnostics;
    NodeTreeReader trees;
    ImageSamples images;
    MaterialReader materials;
    MeshReader meshes;
    StrandReader strands;
    std::set<std::uintptr_t> dirtyOwners;
    std::set<std::uintptr_t> dirtyData;
    std::set<std::uintptr_t> dirtyMaterials;
    //! Blender reuses one temporary object for every dupli, so a dupli is keyed by its original, data and parent instead.
    std::map<std::array<std::uintptr_t, 3>, ObjectState> objects;
    std::unordered_map<std::uintptr_t, std::uintptr_t> parentOriginals;
    LightLinking linking;
    //! The originals of the node trees that changed, embedded or groups.
    std::set<std::uintptr_t> dirtyTrees;
    bool imagesChanged = false;
    std::map<std::pair<std::uintptr_t, std::string>, Object> instanceAttributes;
};

std::expected<void, Core::Error> SceneSync::sync(Scene::Scene& scene, const Object& depsgraph, const bool approximate) {
    ++generation;
    Pass pass(scene, depsgraph, approximate, textures, attributeTables);
    Reader& reader = pass.reader;
    readUpdates(pass, depsgraph);
    //! A new depsgraph lists no updates for its data, and new data can take a freed address, so the records keyed by address go.
    const Source source{reader.pointer(depsgraph), static_cast<std::uint32_t>(reader.integer(reader.get(depsgraph, "scene"), "session_uid"))};
    if (textures.setLimit(textureLimit(reader, depsgraph)))
        pass.imagesChanged = true;
    if (approximate != this->approximate || source != this->source)
        reset(scene);
    this->approximate = approximate;
    this->source = source;
    readEmitters(pass, depsgraph);
    reader.forEach(reader.get(depsgraph, "object_instances"), [&](const Object& instance) { syncInstance(pass, instance); });
    applyLightLinking(pass);
    syncCamera(pass);
    syncWorld(pass);
    warnings = pass.diagnostics.takeWarnings();
    if (std::expected<void, Core::Error> read = reader.takeError(); !read) {
        reset(scene);
        return read;
    }
    removeUnseen(scene);
    removeUnusedTextures(scene);
    return {};
}

void SceneSync::removeUnusedTextures(Scene::Scene& scene) {
    std::set<std::pair<std::uint32_t, std::uint32_t>> used;
    const auto use = [&](const Scene::TextureHandle handle) { used.insert({handle.index, handle.generation}); };
    const auto useAll = [&](const std::vector<Scene::MaterialTexture>& list) {
        for (const Scene::MaterialTexture& texture : list)
            use(texture.texture);
    };
    for (const Scene::Material& material : scene.getMaterials().getValues()) {
        useAll(material.graphTextures);
        useAll(material.displacementTextures);
        for (const std::optional<Scene::MaterialTexture>& texture : material.textures)
            if (texture)
                use(texture->texture);
    }
    for (const Scene::Light& light : scene.getLights().getValues())
        useAll(light.graphTextures);
    for (const Scene::Volume& volume : scene.getVolumes().getValues())
        useAll(volume.densityTextures);
    for (const auto& [original, record] : materials)
        if (record.volume)
            useAll(record.volume->densityTextures);
    for (const auto& [key, record] : objectVolumes)
        if (record.volume)
            useAll(record.volume->densityTextures);
    use(scene.getEnvironment().texture);
    useAll(scene.getEnvironment().graphTextures);
    if (worldSky)
        use(worldSky->second);
    textures.removeUnused(scene, used);
}

void SceneSync::readUpdates(Pass& pass, const Object& depsgraph) {
    Reader& reader = pass.reader;
    reader.forEach(reader.get(depsgraph, "updates"), [&](const Object& update) {
        const Object id = reader.get(update, "id");
        const std::string type = reader.text(id, "id_type");
        const std::uintptr_t original = reader.pointer(reader.get(id, "original"));
        if (type == "IMAGE") {
            textures.forgetImage(static_cast<std::uint32_t>(reader.integer(id, "session_uid")));
            pass.imagesChanged = true;
        } else if (type == "NODETREE")
            pass.dirtyTrees.insert(original);
        else if (type == "MATERIAL")
            pass.dirtyMaterials.insert(original);
        else if (reader.flag(update, "is_updated_geometry"))
            (type == "OBJECT" ? pass.dirtyOwners : pass.dirtyData).insert(original);
    });
}

void SceneSync::readEmitters(Pass& pass, const Object& depsgraph) {
    Reader& reader = pass.reader;
    LightLinking& linking = pass.linking;
    const auto addMembers = [&](Emitter& emitter, const Object& objects, const bool include) {
        reader.forEach(objects, [&](const Object& member) { emitter.members[reader.pointer(reader.get(member, "original"))] = include; });
        emitter.include = emitter.include || include;
    };
    reader.forEach(reader.get(depsgraph, "objects"), [&](const Object& object) {
        const std::string type = reader.text(object, "type");
        if (type != "MESH" && type != "LIGHT")
            return;
        Emitter emitter;
        emitter.volumes = type != "LIGHT" || reader.flag(object, "visible_volume_scatter");
        const Object settings = reader.find(object, "light_linking");
        if (const Object blockers = settings ? reader.find(settings, "blocker_collection") : Object())
            pass.diagnostics.unsupported(std::format("object '{}': shadow linking", reader.text(object, "name")));
        if (const Object receivers = settings ? reader.find(settings, "receiver_collection") : Object()) {
            emitter.linked = true;
            const std::vector<Object> objects = reader.items(reader.get(receivers, "objects"));
            const std::vector<Object> links = reader.items(reader.get(receivers, "collection_objects"));
            for (std::size_t index = 0; index < std::min(objects.size(), links.size()); ++index) {
                const bool include = reader.text(reader.get(reader.get(links[index], "light_linking"), "link_state")) == "INCLUDE";
                emitter.members[reader.pointer(reader.get(objects[index], "original"))] = include;
                emitter.include = emitter.include || include;
            }
            const std::vector<Object> children = reader.items(reader.get(receivers, "children"));
            const std::vector<Object> childLinks = reader.items(reader.get(receivers, "collection_children"));
            for (std::size_t index = 0; index < std::min(children.size(), childLinks.size()); ++index) {
                const bool include = reader.text(reader.get(reader.get(childLinks[index], "light_linking"), "link_state")) == "INCLUDE";
                addMembers(emitter, reader.get(children[index], "all_objects"), include);
            }
        }
        if (!emitter.linked && emitter.volumes)
            return;
        linking.emitterIndices[reader.pointer(reader.get(object, "original"))] = linking.emitters.size();
        linking.emitters.push_back(std::move(emitter));
    });
    if (!linking.emitters.empty())
        lightSet(pass, 0, false);
}

std::uint32_t SceneSync::lightSet(Pass& pass, const std::uintptr_t original, const bool volume) {
    LightLinking& linking = pass.linking;
    if (linking.emitters.empty())
        return 0;
    const auto [cached, insertedReceiver] = linking.receivers.try_emplace({original, volume}, 0);
    if (!insertedReceiver)
        return cached->second;
    std::vector<std::uint32_t> signature;
    for (std::uint32_t index = 0; index < linking.emitters.size(); ++index) {
        const Emitter& emitter = linking.emitters[index];
        if (volume && !emitter.volumes)
            continue;
        const auto member = original != 0 ? emitter.members.find(original) : emitter.members.end();
        if (!emitter.linked || (member != emitter.members.end() ? member->second : !emitter.include))
            signature.push_back(index);
    }
    const auto [found, inserted] = linking.sets.try_emplace(signature, static_cast<std::uint32_t>(linking.signatures.size()));
    if (inserted && linking.signatures.size() == Shaders::lightSetCount) {
        linking.sets.erase(found);
        if (!linking.overflow)
            pass.diagnostics.unsupported(std::format("light linking: more than {} light sets", Shaders::lightSetCount));
        linking.overflow = true;
        return 0;
    }
    if (inserted)
        linking.signatures.push_back(std::move(signature));
    cached->second = found->second;
    return found->second;
}

std::uint64_t SceneSync::lightMask(const Pass& pass, const std::uintptr_t original) const {
    const LightLinking& linking = pass.linking;
    const auto found = linking.emitterIndices.find(original);
    if (found == linking.emitterIndices.end())
        return ~std::uint64_t{0};
    std::uint64_t mask = 0;
    for (std::size_t set = 0; set < linking.signatures.size(); ++set)
        if (std::ranges::binary_search(linking.signatures[set], static_cast<std::uint32_t>(found->second)))
            mask |= std::uint64_t{1} << set;
    return mask;
}

void SceneSync::applyLightLinking(Pass& pass) {
    for (auto& [key, record] : instances) {
        if (record.generation != generation)
            continue;
        const std::uint64_t mask = lightMask(pass, record.original);
        if (mask == record.lightMask)
            continue;
        record.lightMask = mask;
        pass.scene.setInstanceLightLinking(record.handle, record.lightSet, mask);
    }
    for (auto& [key, record] : lights) {
        if (record.generation != generation)
            continue;
        const std::uint64_t mask = lightMask(pass, record.original);
        const Scene::Light* light = pass.scene.find(record.handle);
        if (mask == record.lightMask || !light)
            continue;
        Scene::Light changed = *light;
        changed.lightMask = mask;
        record.lightMask = mask;
        pass.scene.setLight(record.handle, changed);
    }
}

void SceneSync::syncInstance(Pass& pass, const Object& instance) {
    Reader& reader = pass.reader;
    const bool showSelf = reader.flag(instance, "show_self");
    if (!showSelf && !reader.flag(instance, "show_particles"))
        return;
    const Object object = reader.get(instance, "object");
    const bool instanced = reader.flag(instance, "is_instance");
    std::uintptr_t parent = 0;
    if (instanced) {
        const Object parentObject = reader.get(instance, "parent");
        const auto [found, inserted] = pass.parentOriginals.try_emplace(reader.pointer(parentObject));
        if (inserted)
            found->second = reader.pointer(reader.get(parentObject, "original"));
        parent = found->second;
    }
    std::array<std::uintptr_t, 3> objectKey{reader.pointer(object), 0, 0};
    if (instanced) {
        const Object data = reader.get(object, "data");
        objectKey = {reader.pointer(reader.get(object, "original")), data.get() == Py_None ? 0 : reader.pointer(data), parent};
    }
    const auto [found, inserted] = pass.objects.try_emplace(objectKey);
    if (inserted)
        found->second = resolveObject(pass, object, parent);
    const ObjectState& state = found->second;
    if (state.kind == ObjectState::Kind::Skip || reader.failed())
        return;
    InstanceKey key{};
    if (instanced) {
        key[0] = parent;
        const Object persistent = reader.get(instance, "persistent_id");
        const Object values(PySequence_Tuple(persistent.get()));
        if (!values) {
            PyErr_Clear();
            reader.fail("Blender data: persistent_id is not a sequence");
            return;
        }
        const Py_ssize_t count = std::min<Py_ssize_t>(PyTuple_Size(values.get()), 8);
        for (Py_ssize_t index = 0; index < count; ++index)
            key[1 + std::size_t(index)] = static_cast<std::uint32_t>(PyLong_AsLong(PyTuple_GetItem(values.get(), index)));
        key[9] = 1;
    } else {
        key[0] = state.original;
    }
    const Core::Transform transform = toYUp(reader.matrix(reader.get(instance, "matrix_world")));
    if (reader.failed())
        return;
    if (state.kind == ObjectState::Kind::Light)
        placeLight(pass, key, state.original, state.light, transform);
    if (state.kind != ObjectState::Kind::Geometry)
        return;
    const std::uint32_t set = lightSet(pass, state.original, false);
    //! Cycles' random id: the hash of the name, then of each persistent id of an instance.
    std::uint32_t random = CyclesHash::hash(reader.text(object, "name"));
    for (std::size_t index = 1; index < (instanced ? 9u : 2u); ++index)
        random = CyclesHash::hash(random, instanced ? static_cast<std::uint32_t>(key[index]) : 0u);
    const std::vector<Core::float4> attributes = readAttributes(pass, object, instanced ? reader.get(instance, "parent") : Object(), key);
    if (reader.failed())
        return;
    if (showSelf && pass.scene.find(state.mesh))
        placeInstance(pass, key, state.mesh, state, transform, set, CyclesHash::toFloat(random), attributes);
    if (showSelf && state.volume)
        placeVolume(pass, key, *state.volume->volume, state.volume->version, transform, lightSet(pass, state.original, true));
    for (std::size_t index = 0; index < state.strands.size(); ++index) {
        InstanceKey strandKey = key;
        strandKey[9] += 2 * (index + 1);
        placeInstance(pass, strandKey, state.strands[index], state, transform, set, CyclesHash::toFloat(random), attributes);
    }
}

std::vector<Core::float4> SceneSync::readAttributes(Pass& pass, const Object& object, const Object& parent, const InstanceKey& key) {
    Reader& reader = pass.reader;
    std::vector<Core::float4> result;
    const auto toFloat4 = [&](const Object& value) {
        const Values values = reader.values(value);
        return Core::float4{values.values[0], values.values[1], values.values[2], values.values[3]};
    };
    for (const Materials::InstanceAttribute& attribute : attributeTables.instances.getAttributes()) {
        const Object name = reader.string(attribute.name);
        if (attribute.scope == Materials::AttributeScope::ViewLayer) {
            result.push_back(toFloat4(reader.helper("view_layer_attribute", std::array{pass.depsgraph, name})));
            continue;
        }
        if (attribute.scope == Materials::AttributeScope::Instancer && parent) {
            const auto [found, inserted] = pass.instanceAttributes.try_emplace({key[0], attribute.name});
            if (inserted)
                found->second = reader.helper("instance_attribute", std::array{parent, name});
            //! The last persistent id in use is the index of the instance in the outermost geometry.
            std::size_t last = 8;
            while (last > 1 && key[last] == static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()))
                --last;
            if (found->second && found->second.get() != Py_None && key[last] < reader.size(found->second)) {
                result.push_back(toFloat4(reader.at(found->second, key[last])));
                continue;
            }
        }
        result.push_back(
            toFloat4(reader.helper("object_attribute", std::array{attribute.scope == Materials::AttributeScope::Instancer && parent ? parent : object, name})));
    }
    return result;
}

SceneSync::ObjectState SceneSync::resolveObject(Pass& pass, const Object& object, const std::uintptr_t parent) {
    Reader& reader = pass.reader;
    ObjectState state;
    const std::string type = reader.text(object, "type");
    const bool surface = type == "MESH" || type == "CURVES" || type == "POINTCLOUD";
    if (!surface && type != "LIGHT" && type != "VOLUME")
        return state;
    state.original = reader.pointer(reader.get(object, "original"));
    if (type == "LIGHT") {
        state.light = readLight(pass, object);
        state.kind = ObjectState::Kind::Light;
        return state;
    }
    if (surface)
        state.mesh = resolveMesh(pass, object, type, state.original, parent);
    if (type == "MESH" && parent == 0 && reader.size(reader.get(object, "particle_systems")) > 0)
        state.strands = resolveStrands(pass, object, state.original, parent);
    if (type == "MESH" || type == "VOLUME")
        state.volume = resolveVolume(pass, object, state.original, parent);
    state.kind = ObjectState::Kind::Geometry;
    state.flags = Scene::InstanceFlags::DoubleSided;
    if (!reader.flag(object, "visible_camera"))
        state.flags = state.flags | Scene::InstanceFlags::NoCameraRays;
    if (!reader.flag(object, "visible_shadow"))
        state.flags = state.flags | Scene::InstanceFlags::NoShadow;
    if (!reader.flag(object, "visible_diffuse"))
        state.flags = state.flags | Scene::InstanceFlags::NoDiffuseRays;
    if (!reader.flag(object, "visible_glossy"))
        state.flags = state.flags | Scene::InstanceFlags::NoGlossyRays;
    if (!reader.flag(object, "visible_transmission"))
        state.flags = state.flags | Scene::InstanceFlags::NoTransmissionRays;
    if (!reader.flag(object, "visible_volume_scatter"))
        state.flags = state.flags | Scene::InstanceFlags::NoVolumeScatterRays;
    if (reader.flag(object, "is_holdout"))
        state.flags = state.flags | Scene::InstanceFlags::Holdout;
    const Values color = reader.values(object, "color");
    state.color = {color.values[0], color.values[1], color.values[2], color.values[3]};
    state.passIndex = static_cast<std::uint32_t>(reader.integer(object, "pass_index"));
    return state;
}

const SceneSync::MaterialRecord* SceneSync::material(Pass& pass, const Object& material) {
    Reader& reader = pass.reader;
    if (!material) {
        if (!pass.scene.find(defaultMaterial.handle))
            defaultMaterial.handle = pass.scene.addMaterial({.name = "NoorRay Default", .nodeTree = defaultTree()});
        return &defaultMaterial;
    }
    const std::uintptr_t original = reader.pointer(reader.get(material, "original"));
    if (reader.failed())
        return nullptr;
    const auto [found, inserted] = materials.try_emplace(original);
    MaterialRecord& record = found->second;
    const bool treeChanged = std::ranges::any_of(record.trees, [&](const std::uintptr_t tree) { return pass.dirtyTrees.contains(tree); });
    if (record.generation != generation && (inserted || treeChanged || pass.imagesChanged || pass.dirtyMaterials.contains(original))) {
        MaterialReader::Result result = pass.materials.read(material);
        record.trees.clear();
        collectTrees(reader, nodeTree(reader, material), record.trees);
        if (reader.failed())
            return nullptr;
        if (inserted)
            record.handle = pass.scene.addMaterial(std::move(result.material));
        else
            pass.scene.setMaterial(record.handle, result.material);
        record.needs = std::move(result.needs);
        record.volume = std::move(result.volume);
        record.version = ++versions;
    }
    record.generation = generation;
    return &record;
}

bool SceneSync::isStale(const Pass& pass, const std::uintptr_t original, const std::uintptr_t parent, const std::uintptr_t data) const {
    return pass.dirtyOwners.contains(original) || pass.dirtyOwners.contains(parent) || pass.dirtyData.contains(data);
}

const std::vector<Scene::MeshHandle>& SceneSync::resolveStrands(Pass& pass, const Object& object, const std::uintptr_t original, const std::uintptr_t parent) {
    Reader& reader = pass.reader;
    const auto [found, inserted] = strands.try_emplace({original, parent});
    StrandRecord& record = found->second;
    if (record.generation == generation)
        return record.handles;
    record.generation = generation;
    std::vector<Scene::MaterialHandle> slots;
    for (const Object& slot : reader.items(reader.get(object, "material_slots")))
        if (const MaterialRecord* read = material(pass, reader.find(slot, "material")))
            slots.push_back(read->handle);
    if (slots.empty())
        if (const MaterialRecord* read = material(pass, {}))
            slots.push_back(read->handle);
    const std::uintptr_t data = reader.pointer(reader.get(reader.get(object, "data"), "original"));
    if (!inserted && slots == record.materials && !isStale(pass, original, parent, data))
        return record.handles;
    record.materials = slots;
    for (const Scene::MeshHandle handle : record.handles)
        pass.scene.remove(handle);
    record.handles.clear();
    const bool render = reader.text(pass.depsgraph, "mode") == "RENDER";
    for (Scene::Mesh& mesh : pass.strands.readParticles(object, render, slots))
        if (!mesh.indices.empty())
            record.handles.push_back(pass.scene.addMesh(std::move(mesh)));
    return record.handles;
}

const SceneSync::ObjectVolume* SceneSync::resolveVolume(Pass& pass, const Object& object, const std::uintptr_t original, const std::uintptr_t parent) {
    Reader& reader = pass.reader;
    Object slotMaterial;
    const MaterialRecord* read = nullptr;
    for (const Object& slot : reader.items(reader.get(object, "material_slots"))) {
        const Object candidate = reader.find(slot, "material");
        const MaterialRecord* record = candidate ? material(pass, candidate) : nullptr;
        if (!record || !record->volume)
            continue;
        slotMaterial = candidate;
        read = record;
        break;
    }
    if (reader.failed() || !read)
        return nullptr;
    const std::string type = reader.text(object, "type");
    const Object data = reader.get(object, "data");
    const std::uintptr_t dataPointer = reader.pointer(data);
    const bool fluid = type == "MESH" && reader.helper("fluid_domain", std::array{object});
    const Object path = type == "VOLUME" ? reader.helper("volume_path", std::array{data}) : Object();
    const std::string pathText = path ? reader.text(path) : std::string();
    const std::uintptr_t materialOriginal = reader.pointer(reader.get(slotMaterial, "original"));
    const std::uintptr_t dataOriginal = reader.pointer(reader.get(data, "original"));
    const auto [found, inserted] = objectVolumes.try_emplace({original, parent});
    ObjectVolume& record = found->second;
    const bool current = !inserted && record.materialVersion == read->version && record.material == materialOriginal && record.path == pathText
                         && (record.data == dataPointer || fluid) && !isStale(pass, original, parent, dataOriginal);
    if (record.generation != generation && !current) {
        record.materialVersion = read->version;
        record.material = materialOriginal;
        record.data = dataPointer;
        record.path = pathText;
        record.version = ++versions;
        if (type == "VOLUME" || fluid) {
            GridSet grids(reader, pass.diagnostics, fluid ? object : path);
            record.volume = pass.materials.readVolume(*pass.trees.read(nodeTree(reader, slotMaterial)), reader.text(slotMaterial, "name"),
                                                      "ShaderNodeOutputMaterial", &grids);
            if (record.volume && record.volume->density.empty())
                record.volume.reset();
        } else {
            record.volume = read->volume;
            shapeVolume(pass, data, *record.volume);
        }
    }
    record.generation = generation;
    return record.volume && !record.volume->bounds.isEmpty() ? &record : nullptr;
}

void SceneSync::shapeVolume(Pass& pass, const Object& data, Scene::Volume& volume) {
    MeshLayout layout;
    const Scene::Mesh mesh = pass.meshes.read(data, {}, {}, layout);
    volume.bounds = {};
    for (const std::uint32_t index : mesh.indices)
        volume.bounds.grow(mesh.positions[index]);
    if (volume.bounds.isEmpty())
        return;
    const Core::float3 extent = volume.bounds.extent();
    const float tolerance = 1e-4f * Core::maxComponent(extent);
    const auto onSide = [&](const float value, const float low, const float high) { return value - low < tolerance || high - value < tolerance; };
    const bool box = std::ranges::all_of(mesh.indices, [&](const std::uint32_t index) {
        const Core::float3 p = mesh.positions[index];
        return onSide(p.x, volume.bounds.min.x, volume.bounds.max.x) && onSide(p.y, volume.bounds.min.y, volume.bounds.max.y)
               && onSide(p.z, volume.bounds.min.z, volume.bounds.max.z);
    });
    if (box)
        return;
    const float perUnit = maskSide / Core::maxComponent(extent);
    const auto cells = [&](const float side) { return std::max(static_cast<std::uint32_t>(std::ceil(side * perUnit)), 1u); };
    volume.maskResolution = {cells(extent.x), cells(extent.y), cells(extent.z)};
    volume.mask = fillMask(mesh.positions, mesh.indices, volume.bounds, volume.maskResolution);
    if (volume.kind == Scene::VolumeKind::Grid)
        return;
    volume.kind = Scene::VolumeKind::Grid;
    volume.resolution = {1, 1, 1};
    volume.density = {Core::toHalf(1.0f).bits};
}

Scene::MeshHandle SceneSync::resolveMesh(Pass& pass, const Object& object, const std::string& type, const std::uintptr_t original,
                                         const std::uintptr_t parent) {
    Reader& reader = pass.reader;
    Scene::Scene& scene = pass.scene;
    const Object data = reader.get(object, "data");
    const std::uintptr_t dataPointer = reader.pointer(data);
    const std::uintptr_t dataOriginal = reader.pointer(reader.get(data, "original"));
    std::vector<Object> slots;
    for (const Object& slot : reader.items(reader.get(object, "material_slots")))
        slots.push_back(reader.find(slot, "material"));
    if (slots.empty())
        slots.emplace_back();
    MeshKey meshKey{dataPointer};
    MeshRequest request;
    bool hasSurface = false;
    std::vector<std::string> attributes;
    bool displaces = false;
    for (const Object& slot : slots) {
        const MaterialRecord* record = material(pass, slot);
        if (!record)
            return {};
        const Materials::MaterialNeeds& needs = record->needs;
        request.slotMaterials.push_back(record->handle);
        meshKey.push_back(slot ? reader.pointer(reader.get(slot, "original")) : 0);
        hasSurface |= needs.hasSurface;
        request.tangents |= needs.tangents;
        request.colors |= needs.vertexColors;
        displaces |= needs.displaces;
        if (request.tangentMap.empty())
            request.tangentMap = needs.tangentMap;
        for (std::size_t set = 0; set < needs.streams.size(); ++set)
            if (request.streams[set] && needs.streams[set] && request.streams[set] != needs.streams[set])
                pass.diagnostics.unsupported(
                    std::format("object '{}': its materials read the attributes '{}' and '{}' in one UV set, so the second renders wrong",
                                reader.text(object, "name"), request.streams[set]->name, needs.streams[set]->name));
            else if (needs.streams[set])
                request.streams[set] = needs.streams[set];
        attributes.insert(attributes.end(), needs.attributes.begin(), needs.attributes.end());
    }
    const std::string name = reader.text(object, "name");
    if (reader.failed())
        return {};
    if (!hasSurface)
        return {};
    if (displaces) {
        std::ranges::sort(attributes);
        attributes.erase(std::ranges::unique(attributes).begin(), attributes.end());
        request.attributes = std::move(attributes);
    }

    auto foundMesh = meshes.find(meshKey);
    bool moved = false;
    const std::array<std::uintptr_t, 2> owner{original, parent};
    if (const auto previous = meshOwners.find(owner); foundMesh == meshes.end() && previous != meshOwners.end())
        if (auto old = meshes.find(previous->second); old != meshes.end() && old->second.generation != generation && old->second.users == 1) {
            auto node = meshes.extract(old);
            node.key() = meshKey;
            foundMesh = meshes.insert(std::move(node)).position;
            moved = true;
        }
    const bool insertedMesh = foundMesh == meshes.end();
    if (insertedMesh)
        foundMesh = meshes.try_emplace(meshKey).first;
    meshOwners[owner] = meshKey;
    MeshRecord& mesh = foundMesh->second;
    const bool stale = insertedMesh || moved || mesh.request != request || isStale(pass, original, parent, dataOriginal);
    if (mesh.generation != generation && stale) {
        if (mesh.request != request)
            mesh.layout = {};
        Scene::Mesh read = type == "MESH"     ? pass.meshes.read(data, reader.text(data, "name"), request, mesh.layout)
                           : type == "CURVES" ? pass.strands.readCurves(data, request.slotMaterials.front())
                                              : pass.strands.readPoints(data, request.slotMaterials.front());
        if (reader.failed())
            return {};
        if (mesh.layout.reused) {
            const auto fill = [&](Scene::Mesh& target) { return MeshReader::fill(mesh.layout, target); };
            if (!scene.updateMesh(mesh.handle, !mesh.layout.patches.empty(), fill)) {
                reader.fail(std::format("mesh '{}' has the corners of its last read, but not its sizes", name));
                return {};
            }
        } else if (read.indices.empty()) {
            if (!insertedMesh)
                scene.remove(mesh.handle);
            meshes.erase(foundMesh);
            return {};
        } else {
            if (insertedMesh)
                mesh.handle = scene.addMesh(std::move(read));
            else
                scene.setMesh(mesh.handle, std::move(read));
            mesh.request = std::move(request);
        }
    }
    mesh.users = mesh.generation == generation ? mesh.users + 1 : 1;
    mesh.generation = generation;
    return mesh.handle;
}

void SceneSync::placeInstance(Pass& pass, const InstanceKey& key, const Scene::MeshHandle mesh, const ObjectState& state, const Core::Transform& transform,
                              const std::uint32_t lightSet, const float random, const std::vector<Core::float4>& attributes) {
    Scene::Scene& scene = pass.scene;
    auto [foundInstance, insertedInstance] = instances.try_emplace(key);
    InstanceRecord& instance = foundInstance->second;
    if (!insertedInstance && instance.mesh != mesh) {
        scene.remove(instance.handle);
        insertedInstance = true;
    }
    if (insertedInstance) {
        instance.handle = scene.addInstance({.transform = transform,
                                             .previousTransform = transform,
                                             .mesh = mesh,
                                             .flags = state.flags,
                                             .id = static_cast<std::uint32_t>(instances.size()),
                                             .material = {},
                                             .lightSet = lightSet,
                                             .lightMask = ~std::uint64_t{0},
                                             .color = state.color,
                                             .passIndex = state.passIndex,
                                             .random = random,
                                             .attributes = attributes});
        instance.lightMask = ~std::uint64_t{0};
    } else {
        if (instance.transform != transform)
            scene.setInstanceTransform(instance.handle, transform);
        if (instance.flags != state.flags)
            scene.setInstanceFlags(instance.handle, state.flags);
        if (instance.lightSet != lightSet)
            scene.setInstanceLightLinking(instance.handle, lightSet, instance.lightMask);
        if (instance.color != state.color || instance.passIndex != state.passIndex || instance.random != random || instance.attributes != attributes)
            scene.setInstanceInfo(instance.handle, state.color, state.passIndex, random, attributes);
    }
    instance.original = state.original;
    instance.lightSet = lightSet;
    instance.mesh = mesh;
    instance.transform = transform;
    instance.flags = state.flags;
    instance.color = state.color;
    instance.passIndex = state.passIndex;
    instance.random = random;
    instance.attributes = attributes;
    instance.generation = generation;
}

Scene::Light SceneSync::readLight(Pass& pass, const Object& object) {
    Reader& reader = pass.reader;
    const Object light = reader.get(object, "data");
    Scene::Light result;
    result.name = reader.text(object, "name");
    const Object source = nodeTree(reader, light);
    std::shared_ptr<Materials::NodeTree> tree = pass.trees.read(source);
    std::vector<Materials::NodeProperty>& settings = tree->properties;
    settings.push_back(booleanProperty("use_nodes", static_cast<bool>(source)));
    settings.push_back(numberProperty("energy", reader.number(light, "energy")));
    settings.push_back(numberProperty("exposure", reader.find(light, "exposure") ? reader.number(light, "exposure") : 0.0f));
    const Values color = reader.values(light, "color");
    settings.push_back(numbersProperty("color", std::span(color.values).first(color.count)));
    const bool temperature = reader.find(light, "use_temperature") && reader.flag(light, "use_temperature");
    settings.push_back(booleanProperty("use_temperature", temperature));
    if (temperature) {
        const Values kelvin = reader.values(light, "temperature_color");
        settings.push_back(numbersProperty("temperature_color", std::span(kelvin.values).first(kelvin.count)));
    }
    if (reader.find(light, "normalize"))
        settings.push_back(booleanProperty("normalize", reader.flag(light, "normalize")));
    result.hidden = hiddenRays(reader, object, {"visible_diffuse", "visible_glossy", "visible_transmission", "visible_volume_scatter"});
    result.visibleToCamera = reader.flag(object, "visible_camera");
    result.castsShadow = !reader.find(light, "use_shadow") || reader.flag(light, "use_shadow");
    if (const Object cycles = reader.get(light, "cycles"); cycles && reader.find(cycles, "max_bounces"))
        result.maxBounces = static_cast<std::uint32_t>(std::max(reader.integer(cycles, "max_bounces"), 0));
    const std::string type = reader.text(light, "type");
    if (type == "POINT" || type == "SPOT") {
        result.type = type == "POINT" ? Scene::LightType::Point : Scene::LightType::Spot;
        result.radius = reader.number(light, "shadow_soft_size");
    }
    if (type == "SPOT") {
        result.spotAngle = reader.number(light, "spot_size");
        result.spotBlend = reader.number(light, "spot_blend");
    } else if (type == "AREA") {
        const std::string shape = reader.text(light, "shape");
        const bool round = shape == "DISK" || shape == "ELLIPSE";
        result.spread = reader.number(light, "spread");
        result.type = round ? Scene::LightType::Disk : Scene::LightType::Rect;
        result.width = reader.number(light, "size");
        result.height = shape == "SQUARE" || shape == "DISK" ? result.width : reader.number(light, "size_y");
    } else if (type == "SUN") {
        result.type = Scene::LightType::Directional;
        result.angle = reader.number(light, "angle");
    }
    BlenderTextures loader(reader, pass.diagnostics, textures, pass.scene, pass.trees, pass.images);
    Materials::compileLight(*tree, result, pass.diagnostics.forNodes(), loader);
    pass.diagnostics.settle();
    loader.storeRecords();
    result.nodeTree = std::move(tree);
    return result;
}

void SceneSync::placeLight(Pass& pass, const InstanceKey& key, const std::uintptr_t original, Scene::Light light, const Core::Transform& transform) {
    light.transform = transform;
    const auto [found, inserted] = lights.try_emplace(key);
    LightRecord& record = found->second;
    //! Each sync reads the tree again; an equal tree keeps the last one, so that the light does not change.
    if (const Scene::Light* last = inserted ? nullptr : pass.scene.find(record.handle);
        last && last->nodeTree && Materials::isSameTree(*last->nodeTree, *light.nodeTree))
        light.nodeTree = last->nodeTree;
    record.original = original;
    light.lightMask = record.lightMask;
    if (inserted)
        record.handle = pass.scene.addLight(std::move(light));
    else
        pass.scene.setLight(record.handle, light);
    record.generation = generation;
}

void SceneSync::placeVolume(Pass& pass, const InstanceKey& key, const Scene::Volume& source, const std::uint64_t version, const Core::Transform& transform,
                            const std::uint32_t lightSet) {
    const auto [found, inserted] = volumes.try_emplace(key);
    VolumeRecord& record = found->second;
    record.generation = generation;
    if (!inserted && record.transform == transform && record.version == version && record.lightSet == lightSet)
        return;
    Scene::Volume volume = source;
    volume.transform = transform * source.transform;
    volume.lightSet = lightSet;
    if (volume.densityGraph) {
        //! The grid has gridResolution voxels along its longest side in the world, and as many per metre along the others.
        const Core::float3 extent = volume.bounds.max - volume.bounds.min;
        std::array<float, 3> world{};
        for (int axis = 0; axis < 3; ++axis)
            world[axis] = (&extent.x)[axis] * std::hypot(transform.rows[0][axis], transform.rows[1][axis], transform.rows[2][axis]);
        const float perMetre = static_cast<float>(gridResolution) / std::max(std::max(world[0], world[1]), std::max(world[2], 1e-6f));
        const auto voxels = [&](const float size) { return std::clamp(static_cast<std::uint32_t>(std::ceil(size * perMetre)), 4u, gridResolution); };
        volume.resolution = {voxels(world[0]), voxels(world[1]), voxels(world[2])};
    }
    if (inserted)
        record.handle = pass.scene.addVolume(std::move(volume));
    else
        pass.scene.setVolume(record.handle, volume);
    record.transform = transform;
    record.version = version;
    record.lightSet = lightSet;
}

void SceneSync::syncCamera(Pass& pass) {
    Reader& reader = pass.reader;
    const Object scene = reader.get(pass.depsgraph, "scene_eval");
    const Object object = reader.find(scene, "camera");
    Scene::Camera result;
    result.name = "NoorRay Default";
    if (object) {
        const Object camera = reader.get(object, "data");
        const Object render = reader.get(scene, "render");
        const Object dof = reader.get(camera, "dof");
        result.name = reader.text(object, "name");
        const std::string type = reader.text(camera, "type");
        result.type = type == "ORTHO" ? Scene::CameraType::Orthographic : Scene::CameraType::Perspective;
        const float aspect = reader.number(render, "resolution_x") * reader.number(render, "pixel_aspect_x")
                             / std::max(reader.number(render, "resolution_y") * reader.number(render, "pixel_aspect_y"), 1e-6f);
        std::string fit = reader.text(camera, "sensor_fit");
        float size = reader.number(camera, "sensor_width");
        if (fit == "AUTO")
            fit = aspect >= 1.0f ? "HORIZONTAL" : "VERTICAL";
        else if (fit == "VERTICAL")
            size = reader.number(camera, "sensor_height");
        const float width = fit == "HORIZONTAL" ? size : size * aspect;
        const float height = fit == "HORIZONTAL" ? size / aspect : size;
        const float larger = std::max(width, height);
        const Core::Transform world = reader.matrix(reader.get(object, "matrix_world"));
        float focusDistance = reader.number(dof, "focus_distance");
        if (const Object focus = reader.find(dof, "focus_object")) {
            const Core::float3 target = reader.matrix(reader.get(focus, "matrix_world")).getTranslation();
            focusDistance = -world.inverse().transformPoint(target).z;
        }
        result.transform = toYUp(world);
        result.previousTransform = result.transform;
        result.verticalFov = 2.0f * std::atan(0.5f * height / reader.number(camera, "lens"));
        result.orthographicHeight = reader.number(camera, "ortho_scale") * height / larger;
        result.focusDistance = std::max(focusDistance, 1e-4f);
        result.fStop = reader.flag(dof, "use_dof") ? reader.number(dof, "aperture_fstop") : 0.0f;
        result.sensor.width = width / 1000.0f;
        result.sensor.height = height / 1000.0f;
        result.sensor.offset = {reader.number(camera, "shift_x") * larger / 1000.0f, reader.number(camera, "shift_y") * larger / 1000.0f};
        const int blades = reader.integer(dof, "aperture_blades");
        result.blades = blades >= 3 ? static_cast<std::uint32_t>(blades) : 0;
        result.apertureRotation = reader.number(dof, "aperture_rotation");
        result.apertureRatio = reader.number(dof, "aperture_ratio");
        result.pixelAspect = reader.number(render, "pixel_aspect_x") / std::max(reader.number(render, "pixel_aspect_y"), 1e-6f);
        if (type == "PANO")
            readPanorama(reader, camera, result);
        result.near = reader.number(camera, "clip_start");
        result.far = reader.number(camera, "clip_end");
    }
    if (reader.failed())
        return;
    blenderCamera = result;
    if (pass.scene.find(camera)) {
        if (!keepsCamera)
            pass.scene.setCamera(camera, result);
        return;
    }
    camera = pass.scene.addCamera(std::move(result));
    pass.scene.setActiveCamera(camera);
}

void SceneSync::readPanorama(Reader& reader, const Object& camera, Scene::Camera& result) {
    //! Blender 5 moved the panorama from the Cycles settings to the camera.
    const Object settings = reader.find(camera, "panorama_type") ? camera : reader.get(camera, "cycles");
    const std::string type = reader.text(settings, "panorama_type");
    const auto number = [&](const char* name) { return reader.number(settings, name); };
    if (type == "EQUIRECTANGULAR") {
        result.type = Scene::CameraType::Equirectangular;
        result.panorama = {number("longitude_min"), number("longitude_max"), number("latitude_min"), number("latitude_max")};
    } else if (type == "FISHEYE_EQUIDISTANT") {
        result.type = Scene::CameraType::FisheyeEquidistant;
        result.panorama = {number("fisheye_fov")};
    } else if (type == "FISHEYE_EQUISOLID") {
        result.type = Scene::CameraType::FisheyeEquisolid;
        result.panorama = {number("fisheye_fov"), number("fisheye_lens"), result.sensor.width * 1000.0f, result.sensor.height * 1000.0f};
    } else if (type == "FISHEYE_LENS_POLYNOMIAL") {
        result.type = Scene::CameraType::FisheyePolynomial;
        result.panorama = {number("fisheye_fov"),           number("fisheye_polynomial_k4"), result.sensor.width * 1000.0f,   result.sensor.height * 1000.0f,
                           number("fisheye_polynomial_k0"), number("fisheye_polynomial_k1"), number("fisheye_polynomial_k2"), number("fisheye_polynomial_k3")};
    } else if (type == "MIRRORBALL") {
        result.type = Scene::CameraType::MirrorBall;
    } else if (type == "EQUIANGULAR_CUBEMAP_FACE") {
        result.type = Scene::CameraType::EquiangularCubemapFace;
    } else if (type == "CENTRAL_CYLINDRICAL") {
        result.type = Scene::CameraType::CentralCylindrical;
        result.panorama = {number("central_cylindrical_range_u_min"), number("central_cylindrical_range_u_max"), number("central_cylindrical_range_v_min"),
                           number("central_cylindrical_range_v_max"), number("central_cylindrical_radius")};
    } else {
        reader.fail(std::format("camera '{}' has the unknown panorama type {}", result.name, type));
    }
}

void SceneSync::syncWorld(Pass& pass) {
    Reader& reader = pass.reader;
    const Object scene = reader.get(pass.depsgraph, "scene_eval");
    const Object world = reader.find(scene, "world");
    const Object source = world ? nodeTree(reader, world) : Object();
    std::shared_ptr<Materials::NodeTree> tree = pass.trees.read(source);
    std::vector<Materials::NodeProperty>& settings = tree->properties;
    settings.push_back(textProperty("name", world ? reader.text(world, "name") : std::string()));
    settings.push_back(booleanProperty("use_nodes", static_cast<bool>(source)));
    settings.push_back(booleanProperty("film_transparent", reader.flag(reader.get(scene, "render"), "film_transparent")));
    if (const Object visibility = world ? reader.find(world, "cycles_visibility") : Object())
        for (const char* ray : {"camera", "diffuse", "glossy", "transmission", "scatter"})
            settings.push_back(booleanProperty(std::format("cycles_visibility.{}", ray), reader.flag(visibility, ray)));
    if (const Object cycles = world ? reader.find(world, "cycles") : Object())
        settings.push_back(textProperty("sampling_method", reader.text(cycles, "sampling_method")));
    BlenderTextures loader(reader, pass.diagnostics, textures, pass.scene, pass.trees, pass.images,
                           [&](const Object& node) { return worldSkyTable(pass, node); });
    Scene::Environment environment = Materials::compileEnvironment(*tree, pass.scene.getEnvironment(), pass.diagnostics.forNodes(), loader);
    pass.diagnostics.settle();
    loader.storeRecords();
    syncWorldVolume(pass, world, *tree);
    if (reader.failed())
        return;
    if (worldSky && environment.texture != worldSky->second) {
        pass.scene.remove(worldSky->second);
        worldSky.reset();
    }
    const Scene::Environment& last = pass.scene.getEnvironment();
    environment.nodeTree = last.nodeTree && Materials::isSameTree(*last.nodeTree, *tree) ? last.nodeTree : std::move(tree);
    pass.scene.setEnvironment(environment);
}

void SceneSync::syncWorldVolume(Pass& pass, const Object& world, const Materials::NodeTree& tree) {
    std::optional<Scene::Volume> volume =
        world ? pass.materials.readVolume(tree, pass.reader.text(world, "name"), "ShaderNodeOutputWorld", nullptr) : std::nullopt;
    if (pass.reader.failed() || !volume)
        return;
    if (volume->densityGraph)
        pass.diagnostics.unsupported(std::format("world '{}': the density of the volume is linked and renders as 1", pass.reader.text(world, "name")));
    volume->densityGraph = nullptr;
    volume->kind = Scene::VolumeKind::Homogeneous;
    volume->bounds = {{-worldVolumeSize, -worldVolumeSize, -worldVolumeSize}, {worldVolumeSize, worldVolumeSize, worldVolumeSize}};
    const bool same = worldVolume && worldVolume->second.absorption == volume->absorption && worldVolume->second.scattering == volume->scattering
                      && worldVolume->second.emissionRate == volume->emissionRate && worldVolume->second.anisotropy == volume->anisotropy
                      && worldVolume->second.draineAlpha == volume->draineAlpha;
    if (!same)
        worldVolume.emplace(++versions, std::move(*volume));
    placeVolume(pass, {}, worldVolume->second, worldVolume->first, {}, lightSet(pass, 0, true));
}

Scene::TextureHandle SceneSync::worldSkyTable(Pass& pass, const Object& node) {
    const SkySettings settings = readSkySettings(pass.reader, node);
    if (pass.reader.failed())
        return {};
    if (worldSky && worldSky->first == settings)
        return worldSky->second;
    if (worldSky)
        pass.scene.remove(worldSky->second);
    worldSky.emplace(settings, pass.scene.addTexture(makeSkyTexture(settings, "sky " + pass.reader.text(node, "name"))));
    return worldSky->second;
}

void SceneSync::removeUnseen(Scene::Scene& scene) {
    removeOlder(instances, scene, generation);
    removeOlder(lights, scene, generation);
    removeOlder(volumes, scene, generation);
    std::erase_if(objectVolumes, [&](const auto& item) { return item.second.generation != generation; });
    std::erase_if(strands, [&](const auto& item) {
        if (item.second.generation == generation)
            return false;
        for (const Scene::MeshHandle handle : item.second.handles)
            scene.remove(handle);
        return true;
    });
    removeOlder(meshes, scene, generation);
    std::erase_if(meshOwners, [&](const auto& item) { return !meshes.contains(item.second); });
    removeOlder(materials, scene, generation);
}

void SceneSync::reset(Scene::Scene& scene) {
    scene.clear();
    materials.clear();
    defaultMaterial = {};
    meshes.clear();
    meshOwners.clear();
    instances.clear();
    lights.clear();
    volumes.clear();
    objectVolumes.clear();
    strands.clear();
    worldVolume.reset();
    camera = {};
    textures.clear();
    worldSky.reset();
}

}
