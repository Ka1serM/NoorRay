#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "NoorRay/Format/AnimationData.h"
#include "NoorRay/Materials/NodeTree.h"
#include "NoorRay/Scene/Scene.h"

namespace NoorRay::Format {

struct RecordedAnimation {
    Scene::Scene scene;
    AnimationData animation;
};

//! Turns the frames of a live scene into one scene and its animation. Nodes and Gaussian clouds are not recorded.
class AnimationRecorder {
public:
    //! Frames come in ascending order, and the caller takes the change of the scene after each one, so that it names what changed since the last.
    void record(const Scene::Scene& live, int frame);
    RecordedAnimation take(float framesPerSecond);

private:
    struct Channel {
        std::string_view name;
        //! The graph slot or program parameter, or noSlot for a plain property.
        std::uint32_t slot = noSlot;
        std::uint8_t components = 1;
        bool constant = false;
        std::vector<float> frames;
        std::vector<float> values;
    };

    //! An object of the output scene. The output scene never loses objects, so the record index is also its slot and its dense index.
    template <class Object>
    struct Record {
        Core::Handle<Object> live;
        bool present = false;
        std::vector<Channel> channels;
        std::vector<int> keyFrames;
        std::vector<ObjectValue> keyValues;
    };

    template <class Object>
    class Table {
    public:
        using Handle = Core::Handle<Object>;

        std::vector<Record<Object>> records;

        //! The output object of a live one, or an invalid handle.
        Handle find(const Handle live) const {
            const std::uint32_t record = recordOf(live);
            return record == Handle::invalidIndex ? Handle{} : Handle{record, 0};
        }
        std::uint32_t recordOf(Handle live) const;
        void attach(std::uint32_t record, Handle live);
        //! Frees the records whose live object is gone.
        template <class Keep>
        void release(Keep&& keep);
        //! Attaches each new live object to a free record of its identity, or to a new record at the end; returns the records it attached.
        std::vector<std::uint32_t> attachNew(const Core::SlotMap<Object>& live, const Core::SlotMap<Object>& output);

    private:
        std::vector<std::uint32_t> slots;
    };

    struct MeshState {
        int firstFrame = 0;
        std::optional<std::size_t> sequence;
        std::size_t fullFrame = 0;
    };

    static constexpr std::uint32_t noSlot = UINT32_MAX;

    void attachTextures(const Scene::Scene& live);
    void recordMaterials(const Scene::Scene& live, int frame);
    void recordMeshes(const Scene::Scene& live, std::span<const std::uint32_t> newInstances, int frame);
    void recordMesh(std::uint32_t record, const Scene::Mesh& live, int frame);
    void recordInstances(const Scene::Scene& live, int frame);
    void splitInstances(const Scene::Scene& live, std::span<const std::uint32_t> attached);
    void recordLights(const Scene::Scene& live, int frame);
    void recordVolumes(const Scene::Scene& live, int frame);
    void recordCameras(const Scene::Scene& live, int frame);
    void recordEnvironment(const Scene::Scene& live, int frame);

    void sample(std::vector<Channel>& channels, std::string_view name, std::uint32_t slot, std::span<const float> base, std::span<const float> value,
                bool constant, int frame) const;
    template <class Object>
    void sampleProperties(Record<Object>& record, const Object& base, const Object& value, int frame) const;
    void sampleSlots(std::vector<Channel>& channels, std::string_view name, std::span<const Core::float4> base, std::span<const Core::float4> value,
                     int frame) const;
    void sampleParameters(Record<Scene::Material>& record, const Scene::Material& base, const Scene::Material& value, int frame) const;
    template <class Object>
    void addKey(Record<Object>& record, const Object& base, Object value, int frame) const;

    void remap(std::vector<Scene::MaterialTexture>& list) const;
    Scene::Material remapped(Scene::Material material) const;
    Scene::Light remapped(Scene::Light light) const;
    Scene::Volume remapped(Scene::Volume volume) const;
    Scene::Environment remapped(Scene::Environment environment) const;
    Scene::Instance remapped(Scene::Instance instance) const;
    std::vector<Scene::MeshSubset> remapped(std::span<const Scene::MeshSubset> subsets) const;
    Scene::Mesh remapped(Scene::Mesh mesh) const;
    //! A node tree whose texture handles are those of the output scene.
    std::shared_ptr<const Materials::NodeTree> remapped(const std::shared_ptr<const Materials::NodeTree>& tree);

    template <class Object>
    void write(AnimationTarget target, std::span<Record<Object>> records);
    void writeTrack(AnimationTarget target, std::uint32_t index, std::vector<Channel>& channels);

    Scene::Scene output;
    AnimationData animation;
    bool started = false;
    float firstActiveCamera = 0.0f;
    Table<Scene::Texture> textures;
    Table<Scene::Material> materials;
    Table<Scene::Mesh> meshes;
    Table<Scene::Instance> instances;
    Table<Scene::Light> lights;
    Table<Scene::Volume> volumes;
    Table<Scene::Camera> cameras;
    Record<Scene::Environment> environment;
    std::vector<Channel> sceneChannels;
    std::vector<MeshState> meshStates;
    //! The trees of the current frame, whose sources live in the live scene meanwhile.
    std::unordered_map<const Materials::NodeTree*, std::shared_ptr<const Materials::NodeTree>> trees;
};

}
