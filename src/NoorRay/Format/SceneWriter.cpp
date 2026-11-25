#include "NoorRay/Format/SceneWriter.h"

#include <format>
#include <fstream>
#include <ranges>
#include <system_error>

#include "NoorRay/Format/SceneEncoder.h"

namespace NoorRay::Format {

namespace {

template <class Object>
void writeObjects(BinaryWriter& writer, SceneEncoder& encode, const Core::SlotMap<Object>& objects) {
    writer.write(std::uint64_t{objects.size()});
    for (const Object& object : objects.getValues())
        encode(object);
}

//! Parents come before their children and siblings keep their order, so the reader rebuilds the tree in one pass.
void writeNodes(BinaryWriter& writer, SceneEncoder& encode, const Scene::Scene& scene) {
    writer.write(std::uint64_t{scene.getNodes().size()});
    std::vector<std::pair<Scene::NodeHandle, std::uint32_t>> pending;
    for (const Scene::NodeHandle root : std::views::reverse(scene.getRootNodes()))
        pending.emplace_back(root, Scene::NodeHandle::invalidIndex);
    for (std::uint32_t index = 0; !pending.empty(); ++index) {
        const auto [handle, parent] = pending.back();
        pending.pop_back();
        const Scene::Node& node = *scene.find(handle);
        encode(node.name, node.localTransform, parent, static_cast<std::uint8_t>(node.object.index()));
        std::visit(
            [&]<class Object>(const Object& object) {
                if constexpr (!std::is_same_v<Object, std::monostate>)
                    encode(object);
            },
            node.object);
        for (const Scene::NodeHandle child : std::views::reverse(node.children))
            pending.emplace_back(child, index);
    }
}

void writeScene(BinaryWriter& writer, SceneEncoder& encode, const Scene::Scene& scene) {
    writeObjects(writer, encode, scene.getTextures());
    writeObjects(writer, encode, scene.getMaterials());
    writeObjects(writer, encode, scene.getMeshes());
    writeObjects(writer, encode, scene.getInstances());
    writeObjects(writer, encode, scene.getGaussianClouds());
    writeObjects(writer, encode, scene.getVolumes());
    writeObjects(writer, encode, scene.getLights());
    writeObjects(writer, encode, scene.getCameras());
    encode(scene.getActiveCamera(), scene.getEnvironment());
    writeNodes(writer, encode, scene);
}

template <class Object>
FramePlace writeFrame(BinaryWriter& writer, SceneEncoder& encode, const Object& object) {
    writer.align();
    const std::uint64_t offset = writer.getPosition();
    encode(object);
    return {offset, writer.getPosition() - offset};
}

template <class Object>
std::optional<ObjectValue> copyAt(const Core::SlotMap<Object>& objects, const std::uint32_t index) {
    if (index >= objects.size())
        return std::nullopt;
    return objects.getValues()[index];
}

std::optional<ObjectValue> objectAt(const Scene::Scene& scene, const AnimationTarget target, const std::uint32_t index) {
    switch (target) {
    case AnimationTarget::Scene:
        return std::nullopt;
    case AnimationTarget::Instance:
        return copyAt(scene.getInstances(), index);
    case AnimationTarget::Light:
        return copyAt(scene.getLights(), index);
    case AnimationTarget::Camera:
        return copyAt(scene.getCameras(), index);
    case AnimationTarget::Material:
        return copyAt(scene.getMaterials(), index);
    case AnimationTarget::Environment:
        return scene.getEnvironment();
    case AnimationTarget::Volume:
        return copyAt(scene.getVolumes(), index);
    }
    return std::nullopt;
}

FramePlace writeValue(BinaryWriter& writer, SceneEncoder& encode, const ObjectValue& value) {
    return std::visit([&](const auto& object) { return writeFrame(writer, encode, object); }, value);
}

//! The scene holds the value before the first key, so a late first key gets one in front of it.
std::expected<ObjectKeyDirectory, Core::Error> writeObjectKeys(BinaryWriter& writer, SceneEncoder& encode, const Scene::Scene& scene, const int firstFrame,
                                                               const ObjectKeys& keys) {
    const std::optional<ObjectValue> base = objectAt(scene, keys.target, keys.index);
    if (!base || keys.frames.empty() || keys.frames.size() != keys.values.size()
        || std::ranges::any_of(keys.values, [&](const ObjectValue& value) { return value.index() != base->index(); }))
        return Core::fail(Core::ErrorCode::InvalidArgument, "the object keys of an animation do not match their target");
    ObjectKeyDirectory directory{keys.target, keys.index, {}, {}};
    if (keys.frames.front() > firstFrame) {
        directory.frames.push_back(firstFrame);
        directory.places.push_back(writeValue(writer, encode, *base));
    }
    for (std::size_t key = 0; key < keys.frames.size(); ++key) {
        directory.frames.push_back(keys.frames[key]);
        directory.places.push_back(writeValue(writer, encode, keys.values[key]));
    }
    return directory;
}

MeshSequenceDirectory writeMeshSequence(BinaryWriter& writer, SceneEncoder& encode, const MeshSequence& sequence) {
    MeshSequenceDirectory directory{sequence.mesh, {}, {}, {}};
    for (const MeshFrame& frame : sequence.frames) {
        directory.frames.push_back(frame.frame);
        if (frame.mesh) {
            directory.fullFrames.push_back(frame.frame);
            directory.places.push_back(writeFrame(writer, encode, *frame.mesh));
            continue;
        }
        writer.align();
        const std::uint64_t offset = writer.getPosition();
        encode(frame.positions, frame.normals, frame.tangents);
        directory.places.push_back({offset, writer.getPosition() - offset});
    }
    return directory;
}

std::expected<AnimationDirectory, Core::Error> writeFrames(BinaryWriter& writer, SceneEncoder& encode, const Scene::Scene& scene,
                                                           const AnimationData& animation) {
    AnimationDirectory directory{animation.firstFrame, animation.lastFrame, animation.framesPerSecond, animation.tracks, {}, {}};
    for (const ObjectKeys& keys : animation.objectKeys) {
        std::expected<ObjectKeyDirectory, Core::Error> written = writeObjectKeys(writer, encode, scene, animation.firstFrame, keys);
        if (!written)
            return std::unexpected(std::move(written.error()));
        directory.objectKeys.push_back(std::move(*written));
    }
    for (const MeshSequence& sequence : animation.meshSequences)
        directory.meshSequences.push_back(writeMeshSequence(writer, encode, sequence));
    return directory;
}

std::expected<void, Core::Error> writeFile(const Scene::Scene& scene, const AnimationData& animation, std::ostream& stream) {
    BinaryWriter writer(stream);
    FileHeader header;
    writer.write(header);
    SharedObjects shared;
    SceneEncoder encode(writer, scene, shared);
    writeScene(writer, encode, scene);
    AnimationDirectory directory;
    if (!animation.isEmpty()) {
        std::expected<AnimationDirectory, Core::Error> written = writeFrames(writer, encode, scene, animation);
        if (!written)
            return std::unexpected(std::move(written.error()));
        directory = std::move(*written);
    }
    writer.align();
    header.sharedOffset = writer.getPosition();
    shared.write(writer);
    if (!animation.isEmpty()) {
        writer.align();
        header.animationOffset = writer.getPosition();
        encode(directory);
    }
    header.size = writer.getPosition();
    stream.seekp(0);
    BinaryWriter(stream).write(header);
    return {};
}

std::expected<void, Core::Error> checkTextures(const Scene::Scene& scene) {
    for (const Scene::Texture& texture : scene.getTextures().getValues())
        if (texture.data.empty() && !texture.levels.empty())
            return Core::fail(Core::ErrorCode::InvalidState, std::format("the texels of the texture '{}' are released, so it cannot be saved", texture.name));
    return {};
}

}

std::expected<void, Core::Error> SceneWriter::write(const Scene::Scene& scene, const std::filesystem::path& path) {
    return write(scene, AnimationData{}, path);
}

//! An open scene file stays mapped for its frames, so a save never truncates the file but replaces it.
std::expected<void, Core::Error> SceneWriter::write(const Scene::Scene& scene, const AnimationData& animation, const std::filesystem::path& path) {
    if (std::expected<void, Core::Error> checked = checkTextures(scene); !checked)
        return checked;
    std::filesystem::path partial = path;
    partial += ".partial";
    std::ofstream stream(partial, std::ios::binary | std::ios::trunc);
    if (!stream)
        return Core::fail(Core::ErrorCode::Io, std::format("cannot open '{}' for writing", partial.string()));
    std::expected<void, Core::Error> written = writeFile(scene, animation, stream);
    stream.close();
    if (written && !stream)
        written = Core::fail(Core::ErrorCode::Io, std::format("cannot write '{}'", partial.string()));
    std::error_code error;
    if (written) {
        std::filesystem::rename(partial, path, error);
        if (!error)
            return {};
        written = Core::fail(Core::ErrorCode::Io, std::format("cannot replace '{}': {}", path.string(), error.message()));
    }
    std::filesystem::remove(partial, error);
    return written;
}

}
