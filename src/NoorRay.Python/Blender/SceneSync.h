#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "NoorRay.Python/Blender/Diagnostics.h"
#include "NoorRay.Python/Blender/MaterialReader.h"
#include "NoorRay.Python/Blender/MeshReader.h"
#include "NoorRay.Python/Blender/Reader.h"
#include "NoorRay.Python/Blender/StrandReader.h"
#include "NoorRay.Python/Blender/TextureTable.h"
#include "NoorRay/Core/Error.h"
#include "NoorRay/Scene/Scene.h"

namespace NoorRay::Python {

//! Keeps a NoorRay scene equal to a Blender depsgraph: it reads meshes and materials again only when the depsgraph updates name them.
class SceneSync {
public:
    [[nodiscard]] std::expected<void, Core::Error> sync(Scene::Scene& scene, const Object& depsgraph, bool approximate);
    std::vector<std::string> takeWarnings() { return std::move(warnings); }
    //! The viewport sets the camera itself, so later syncs keep it and do not restart the accumulation twice.
    void keepCamera() { keepsCamera = true; }
    //! The camera of the Blender scene, also while the viewport keeps its own.
    const Scene::Camera& getCamera() const { return blenderCamera; }

private:
    //! The object, or for an instance the instancer, the persistent id within it and a one.
    using InstanceKey = std::array<std::uintptr_t, 10>;
    //! The evaluated mesh and the original materials of its slots.
    using MeshKey = std::vector<std::uintptr_t>;

    struct MaterialRecord {
        Scene::MaterialHandle handle;
        Materials::MaterialNeeds needs;
        std::optional<Scene::Volume> volume;
        //! The originals of the node trees the material reads, so that a change of one of them reads the material again.
        std::vector<std::uintptr_t> trees;
        //! Counts the reads of the material, so that a volume of it knows when to change.
        std::uint64_t version = 0;
        std::uint64_t generation = 0;
    };
    struct MeshRecord {
        Scene::MeshHandle handle;
        MeshRequest request;
        MeshLayout layout;
        //! The objects that used the mesh in its last generation.
        std::uint32_t users = 0;
        std::uint64_t generation = 0;
    };
    struct InstanceRecord {
        Scene::InstanceHandle handle;
        Scene::MeshHandle mesh;
        Core::Transform transform;
        Scene::InstanceFlags flags = Scene::InstanceFlags::None;
        Core::float4 color{1.0f, 1.0f, 1.0f, 1.0f};
        std::uint32_t passIndex = 0;
        float random = 0.0f;
        std::vector<Core::float4> attributes;
        std::uintptr_t original = 0;
        std::uint32_t lightSet = 0;
        std::uint64_t lightMask = ~std::uint64_t{0};
        std::uint64_t generation = 0;
    };
    struct LightRecord {
        Scene::LightHandle handle;
        std::uintptr_t original = 0;
        std::uint64_t lightMask = ~std::uint64_t{0};
        std::uint64_t generation = 0;
    };
    //! The volume of an object from its material and its grids or its shape.
    struct ObjectVolume {
        std::optional<Scene::Volume> volume;
        std::uint64_t materialVersion = 0;
        std::uintptr_t material = 0;
        std::uintptr_t data = 0;
        std::string path;
        std::uint64_t version = 0;
        std::uint64_t generation = 0;
    };
    struct StrandRecord {
        std::vector<Scene::MeshHandle> handles;
        std::vector<Scene::MaterialHandle> materials;
        std::uint64_t generation = 0;
    };
    struct VolumeRecord {
        Scene::VolumeHandle handle;
        Core::Transform transform;
        std::uint64_t version = 0;
        std::uint32_t lightSet = 0;
        std::uint64_t generation = 0;
    };
    //! An emitter with a receiver collection, or a light that volumes do not see, as Cycles' light linking.
    struct Emitter {
        //! The original objects of the receiver collection, true where they are included.
        std::unordered_map<std::uintptr_t, bool> members;
        bool linked = false;
        bool include = false;
        bool volumes = true;
    };
    struct LightLinking;
    //! What every instance of one evaluated object shares within a sync.
    struct ObjectState {
        enum class Kind : std::uint8_t { Skip, Geometry, Light };
        Kind kind = Kind::Skip;
        std::uintptr_t original = 0;
        Scene::MeshHandle mesh;
        //! The meshes of the particle hair.
        std::vector<Scene::MeshHandle> strands;
        Scene::InstanceFlags flags = Scene::InstanceFlags::None;
        //! The Object Info node: the object colour and pass index.
        Core::float4 color{1.0f, 1.0f, 1.0f, 1.0f};
        std::uint32_t passIndex = 0;
        Scene::Light light;
        //! The volume that fills a mesh, or the grids of a volume object or fluid domain; its transform leads to the object.
        const ObjectVolume* volume = nullptr;
    };
    struct Pass;
    using Source = std::pair<std::uintptr_t, std::uint32_t>;

    void readUpdates(Pass& pass, const Object& depsgraph);
    void removeUnusedTextures(Scene::Scene& scene);
    void readEmitters(Pass& pass, const Object& depsgraph);
    //! The light set of a receiver: one set per distinct group of linked emitters that reach it; set 0 holds the receivers outside every collection.
    std::uint32_t lightSet(Pass& pass, std::uintptr_t original, bool volume);
    std::uint64_t lightMask(const Pass& pass, std::uintptr_t original) const;
    void applyLightLinking(Pass& pass);
    void syncInstance(Pass& pass, const Object& instance);
    ObjectState resolveObject(Pass& pass, const Object& object, std::uintptr_t parent);
    const MaterialRecord* material(Pass& pass, const Object& material);
    //! The mesh of the surfaces, curves or points of the object, or none.
    Scene::MeshHandle resolveMesh(Pass& pass, const Object& object, const std::string& type, std::uintptr_t original, std::uintptr_t parent);
    //! The meshes of the particle hair of an object, which are read again when the object or a material changes.
    const std::vector<Scene::MeshHandle>& resolveStrands(Pass& pass, const Object& object, std::uintptr_t original, std::uintptr_t parent);
    const ObjectVolume* resolveVolume(Pass& pass, const Object& object, std::uintptr_t original, std::uintptr_t parent);
    //! Gives a volume the bounds of the mesh that it fills, and the filled share of their cells when the mesh is not a box.
    void shapeVolume(Pass& pass, const Object& data, Scene::Volume& volume);
    bool isStale(const Pass& pass, std::uintptr_t original, std::uintptr_t parent, std::uintptr_t data) const;
    void placeInstance(Pass& pass, const InstanceKey& key, Scene::MeshHandle mesh, const ObjectState& state, const Core::Transform& transform,
                       std::uint32_t lightSet, float random, const std::vector<Core::float4>& attributes);
    //! The values of the instance attributes that the materials read; parent is the instancer of an instance.
    std::vector<Core::float4> readAttributes(Pass& pass, const Object& object, const Object& parent, const InstanceKey& key);
    Scene::Light readLight(Pass& pass, const Object& object);
    void placeLight(Pass& pass, const InstanceKey& key, std::uintptr_t original, Scene::Light light, const Core::Transform& transform);
    void placeVolume(Pass& pass, const InstanceKey& key, const Scene::Volume& source, std::uint64_t version, const Core::Transform& transform,
                     std::uint32_t lightSet);
    void syncCamera(Pass& pass);
    void readPanorama(Reader& reader, const Object& camera, Scene::Camera& result);
    void syncWorld(Pass& pass);
    void syncWorldVolume(Pass& pass, const Object& world, const Materials::NodeTree& tree);
    //! The sky table of the world; a new setting replaces the last table, so an animated sun keeps one table.
    Scene::TextureHandle worldSkyTable(Pass& pass, const Object& node);
    void removeUnseen(Scene::Scene& scene);
    void reset(Scene::Scene& scene);

    std::uint64_t generation = 0;
    //! Materials read in one mode do not hold in the other.
    bool approximate = false;
    //! The depsgraph and the session_uid of its scene, which Blender never gives again in a session, of the last sync.
    Source source;
    std::map<std::uintptr_t, MaterialRecord> materials;
    MaterialRecord defaultMaterial;
    std::map<MeshKey, MeshRecord> meshes;
    //! The mesh of each object and dupli parent in the last pass, so that a modifier result, which is new data at each update, keeps its mesh.
    std::map<std::array<std::uintptr_t, 2>, MeshKey> meshOwners;
    std::map<InstanceKey, InstanceRecord> instances;
    std::map<InstanceKey, LightRecord> lights;
    std::map<InstanceKey, VolumeRecord> volumes;
    std::map<std::array<std::uintptr_t, 2>, ObjectVolume> objectVolumes;
    std::map<std::array<std::uintptr_t, 2>, StrandRecord> strands;
    std::optional<std::pair<std::uint64_t, Scene::Volume>> worldVolume;
    std::uint64_t versions = 0;
    Materials::AttributeTables attributeTables;
    Scene::CameraHandle camera;
    Scene::Camera blenderCamera;
    bool keepsCamera = false;
    TextureTable textures;
    std::optional<std::pair<SkySettings, Scene::TextureHandle>> worldSky;
    std::vector<std::string> warnings;
};

}
