#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "NoorRay/Format/BinaryWriter.h"
#include "NoorRay/Format/SceneCodec.h"

namespace NoorRay::Format {

//! The graphs, programs and node trees that scene objects share, each stored once as its encoded bytes; equal bytes are one object.
class SharedObjects {
public:
    struct Table {
        std::unordered_map<const void*, std::uint32_t> indices;
        std::unordered_map<std::string, std::uint32_t> indicesByBytes;

        std::uint32_t add(const void* object, std::string bytes);
    };

    Table graphs;
    Table programs;
    Table nodeTrees;

    void write(BinaryWriter& writer) const;
};

//! Writes scene objects: handles as dense indices of the scene, shared objects as indices into SharedObjects.
class SceneEncoder {
public:
    SceneEncoder(BinaryWriter& writer, const Scene::Scene& scene, SharedObjects& shared) : writer(writer), scene(scene), shared(shared) {}

    template <class... Values>
    void operator()(const Values&... values) {
        (encode(values), ...);
    }
    void transferSize(const std::size_t size) { writer.write(std::uint64_t{size}); }
    void transferTopology(const Scene::Mesh& mesh);

private:
    template <class T>
    void encode(const T& value) {
        if constexpr (StoredAsBytes<T>)
            writer.write(value);
        else
            transfer(*this, value);
    }
    void encode(const std::string& text) { writer.write(std::string_view(text)); }
    template <class T>
    void encode(const std::span<const T> values) {
        if constexpr (StoredAsBytes<T>) {
            writer.writeArray(values);
        } else {
            writer.write(std::uint64_t{values.size()});
            for (const T& value : values)
                encode(value);
        }
    }
    template <class T>
    void encode(const std::vector<T>& values) {
        encode(std::span<const T>(values));
    }
    template <class T, std::size_t count>
    void encode(const std::array<T, count>& values) {
        for (const T& value : values)
            encode(value);
    }
    template <class T>
    void encode(const std::optional<T>& value) {
        writer.write(value.has_value());
        if (value)
            encode(*value);
    }
    void encode(const Scene::TextureHandle handle) { writer.write(scene.getTextures().denseIndex(handle)); }
    void encode(const Scene::MaterialHandle handle) { writer.write(scene.getMaterials().denseIndex(handle)); }
    void encode(const Scene::MeshHandle handle) { writer.write(scene.getMeshes().denseIndex(handle)); }
    void encode(const Scene::InstanceHandle handle) { writer.write(scene.getInstances().denseIndex(handle)); }
    void encode(const Scene::LightHandle handle) { writer.write(scene.getLights().denseIndex(handle)); }
    void encode(const Scene::GaussianCloudHandle handle) { writer.write(scene.getGaussianClouds().denseIndex(handle)); }
    void encode(const Scene::VolumeHandle handle) { writer.write(scene.getVolumes().denseIndex(handle)); }
    void encode(const Scene::CameraHandle handle) { writer.write(scene.getCameras().denseIndex(handle)); }
    void encode(const std::shared_ptr<const Materials::Graph>& graph);
    void encode(const std::shared_ptr<const Materials::Program>& program);
    void encode(const std::shared_ptr<const Materials::NodeTree>& tree);
    template <class Object, class Source>
    void encodeShared(SharedObjects::Table& table, const Object* object, const Source& getSource);

    BinaryWriter& writer;
    const Scene::Scene& scene;
    SharedObjects& shared;
};

}
