#include "NoorRay/Format/SceneFile.h"

#include <format>

#include "NoorRay/Format/SceneDecoder.h"

namespace NoorRay::Format {

namespace {

template <class Object>
using AddObject = Core::Handle<Object> (Scene::Scene::*)(Object);

template <class Object>
std::expected<void, Core::Error> loadObjects(BinaryReader& reader, SceneIndex& index, Scene::Scene& scene, const AddObject<Object> add,
                                             const std::string_view part) {
    const std::uint64_t count = reader.readCount();
    for (std::uint64_t entry = 0; entry < count; ++entry) {
        std::expected<Object, Core::Error> object = SceneDecoder(reader, index).read<Object>(part);
        if (!object)
            return std::unexpected(std::move(object.error()));
        index.get<Object>().push_back((scene.*add)(std::move(*object)));
    }
    return reader.check(part);
}

template <class Object>
std::expected<Scene::NodeObject, Core::Error> readNodeObject(BinaryReader& reader, const SceneIndex& index) {
    std::expected<Core::Handle<Object>, Core::Error> handle = SceneDecoder(reader, index).read<Core::Handle<Object>>("node list");
    if (!handle)
        return std::unexpected(std::move(handle.error()));
    return Scene::NodeObject{*handle};
}

std::expected<Scene::NodeObject, Core::Error> readNodeObject(BinaryReader& reader, const SceneIndex& index, const std::uint8_t kind) {
    switch (kind) {
    case 0:
        return Scene::NodeObject{};
    case 1:
        return readNodeObject<Scene::Instance>(reader, index);
    case 2:
        return readNodeObject<Scene::Light>(reader, index);
    case 3:
        return readNodeObject<Scene::GaussianCloud>(reader, index);
    case 4:
        return readNodeObject<Scene::Volume>(reader, index);
    case 5:
        return readNodeObject<Scene::Camera>(reader, index);
    default:
        return Core::fail(Core::ErrorCode::InvalidFile, "the scene file has a node with an unknown object kind");
    }
}
static_assert(std::variant_size_v<Scene::NodeObject> == 6);

std::expected<void, Core::Error> loadNodes(BinaryReader& reader, const SceneIndex& index, Scene::Scene& scene) {
    std::vector<Scene::NodeHandle> nodes;
    const std::uint64_t count = reader.readCount();
    for (std::uint64_t entry = 0; entry < count; ++entry) {
        Scene::Node node;
        reader.read(node.name);
        reader.read(node.localTransform);
        const std::uint32_t parent = reader.read<std::uint32_t>();
        const std::uint8_t kind = reader.read<std::uint8_t>();
        std::expected<Scene::NodeObject, Core::Error> object = readNodeObject(reader, index, kind);
        if (!object)
            return std::unexpected(std::move(object.error()));
        if (std::expected<void, Core::Error> checked = reader.check("node list"); !checked)
            return checked;
        node.object = *object;
        if (parent != Scene::NodeHandle::invalidIndex) {
            if (parent >= nodes.size())
                return Core::fail(Core::ErrorCode::InvalidFile, "the scene file has a node before its parent");
            node.parent = nodes[parent];
        }
        const Scene::NodeHandle handle = scene.addNode(std::move(node));
        if (!handle)
            return Core::fail(Core::ErrorCode::InvalidFile, "the scene file gives one object to two nodes");
        nodes.push_back(handle);
    }
    return reader.check("node list");
}

template <class Object>
std::expected<std::shared_ptr<const Object>, Core::Error> openShared(BinaryReader& reader, const SceneIndex& index, const std::string_view part) {
    std::expected<typename Object::Source, Core::Error> source = SceneDecoder(reader, index).read<typename Object::Source>(part);
    if (!source)
        return std::unexpected(std::move(source.error()));
    auto object = std::make_shared<Object>();
    if (std::expected<void, Core::Error> opened = object->open(std::move(*source), part); !opened)
        return std::unexpected(std::move(opened.error()));
    return object;
}

template <>
std::expected<std::shared_ptr<const Materials::NodeTree>, Core::Error> openShared(BinaryReader& reader, const SceneIndex& index, const std::string_view part) {
    std::expected<Materials::NodeTree, Core::Error> tree = SceneDecoder(reader, index).read<Materials::NodeTree>(part);
    if (!tree)
        return std::unexpected(std::move(tree.error()));
    return std::make_shared<Materials::NodeTree>(std::move(*tree));
}

template <class Object>
std::expected<void, Core::Error> loadShared(BinaryReader& reader, SceneIndex& index, const std::string_view part) {
    const std::uint64_t count = reader.readCount();
    for (std::uint64_t entry = 0; entry < count && reader.isValid(); ++entry) {
        BinaryReader object(reader.takeArray(1));
        std::expected<std::shared_ptr<const Object>, Core::Error> opened = openShared<Object>(object, index, part);
        if (!opened)
            return std::unexpected(std::move(opened.error()));
        index.getShared<Object>().push_back(std::move(*opened));
    }
    return reader.check(part);
}

std::expected<void, Core::Error> loadScene(BinaryReader& reader, SceneIndex& index, Scene::Scene& scene) {
    if (auto loaded = loadObjects<Scene::Material>(reader, index, scene, &Scene::Scene::addMaterial, "material list"); !loaded)
        return loaded;
    if (auto loaded = loadObjects<Scene::Mesh>(reader, index, scene, &Scene::Scene::addMesh, "mesh list"); !loaded)
        return loaded;
    if (auto loaded = loadObjects<Scene::Instance>(reader, index, scene, &Scene::Scene::addInstance, "instance list"); !loaded)
        return loaded;
    if (auto loaded = loadObjects<Scene::GaussianCloud>(reader, index, scene, &Scene::Scene::addGaussianCloud, "Gaussian cloud list"); !loaded)
        return loaded;
    if (auto loaded = loadObjects<Scene::Volume>(reader, index, scene, &Scene::Scene::addVolume, "volume list"); !loaded)
        return loaded;
    if (auto loaded = loadObjects<Scene::Light>(reader, index, scene, &Scene::Scene::addLight, "light list"); !loaded)
        return loaded;
    if (auto loaded = loadObjects<Scene::Camera>(reader, index, scene, &Scene::Scene::addCamera, "camera list"); !loaded)
        return loaded;
    std::expected<Scene::CameraHandle, Core::Error> activeCamera = SceneDecoder(reader, index).read<Scene::CameraHandle>("active camera");
    if (!activeCamera)
        return std::unexpected(std::move(activeCamera.error()));
    if (*activeCamera)
        scene.setActiveCamera(*activeCamera);
    std::expected<Scene::Environment, Core::Error> environment = SceneDecoder(reader, index).read<Scene::Environment>("environment");
    if (!environment)
        return std::unexpected(std::move(environment.error()));
    scene.setEnvironment(*environment);
    return loadNodes(reader, index, scene);
}

std::expected<void, Core::Error> loadSharedObjects(BinaryReader& reader, SceneIndex& index) {
    if (auto loaded = loadShared<Materials::Graph>(reader, index, "graph list"); !loaded)
        return loaded;
    if (auto loaded = loadShared<Materials::Program>(reader, index, "program list"); !loaded)
        return loaded;
    return loadShared<Materials::NodeTree>(reader, index, "node tree list");
}

}

std::expected<void, Core::Error> SceneFile::open(const std::filesystem::path& path) {
    file = std::make_shared<Core::MappedFile>();
    if (std::expected<void, Core::Error> opened = file->open(path); !opened)
        return opened;
    const std::span<const std::byte> bytes = file->getBytes();
    BinaryReader reader(bytes);
    reader.read(header);
    if (!reader.isValid() || header.magic != fileMagic)
        return Core::fail(Core::ErrorCode::InvalidFile, std::format("'{}' is not a NoorRay scene (NRSC)", path.string()));
    if (header.version != fileVersion)
        return Core::fail(Core::ErrorCode::InvalidFile,
                          std::format("'{}' has the scene format version {}, not {}", path.string(), header.version, fileVersion));
    const auto isPlace = [&](const std::uint64_t offset) { return offset % arrayAlignment == 0 && offset <= bytes.size(); };
    if (header.size != bytes.size() || header.sharedOffset < sizeof(FileHeader) || !isPlace(header.sharedOffset) || !isPlace(header.animationOffset))
        return Core::fail(Core::ErrorCode::InvalidFile, std::format("'{}' is cut short or damaged", path.string()));
    return {};
}

//! Node trees reference textures, so the textures load first, then the shared objects, then the rest of the scene.
std::expected<Animation, Core::Error> SceneFile::load(Scene::Scene& scene) const {
    const std::span<const std::byte> bytes = file->getBytes();
    SceneIndex index;
    BinaryReader reader(bytes.first(header.sharedOffset));
    reader.read<FileHeader>();
    if (auto loaded = loadObjects<Scene::Texture>(reader, index, scene, &Scene::Scene::addTexture, "texture list"); !loaded)
        return std::unexpected(std::move(loaded.error()));
    BinaryReader sharedReader(bytes.subspan(header.sharedOffset));
    if (auto loaded = loadSharedObjects(sharedReader, index); !loaded)
        return std::unexpected(std::move(loaded.error()));
    if (auto loaded = loadScene(reader, index, scene); !loaded)
        return std::unexpected(std::move(loaded.error()));
    Animation animation;
    if (header.animationOffset == 0)
        return animation;
    BinaryReader animationReader(bytes.subspan(header.animationOffset));
    std::expected<AnimationDirectory, Core::Error> directory = SceneDecoder(animationReader, index).read<AnimationDirectory>("animation");
    if (!directory)
        return std::unexpected(std::move(directory.error()));
    if (auto opened = animation.open(*directory, std::move(index), file, scene); !opened)
        return std::unexpected(std::move(opened.error()));
    return animation;
}

}
