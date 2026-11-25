#include "NoorRay/Scene/Scene.h"

#include <algorithm>
#include <type_traits>
#include <utility>

#include "NoorRay/Materials/Graph.h"

namespace NoorRay::Scene {

namespace {

std::uint64_t objectKey(const NodeObject& object) {
    const std::uint32_t slot = std::visit(
        []<class Object>(const Object& handle) {
            if constexpr (std::is_same_v<Object, std::monostate>)
                return NodeHandle::invalidIndex;
            else
                return handle.index;
        },
        object);
    return (std::uint64_t{object.index()} << 32) | slot;
}

}

MeshHandle Scene::addMesh(Mesh mesh) {
    if (mesh.bounds.isEmpty())
        mesh.computeBounds();
    change.mark(Change::MeshTopology);
    return meshes.insert(std::move(mesh));
}

MaterialHandle Scene::addMaterial(Material material) {
    change.mark(Change::MaterialSet);
    return materials.insert(std::move(material));
}

TextureHandle Scene::addTexture(Texture texture) {
    change.mark(Change::Textures);
    return textures.insert(std::move(texture));
}

InstanceHandle Scene::addInstance(Instance instance) {
    instance.previousTransform = instance.transform;
    change.mark(Change::InstanceSet);
    return instances.insert(std::move(instance));
}

LightHandle Scene::addLight(Light light) {
    change.mark(Change::Lights);
    return lights.insert(std::move(light));
}

GaussianCloudHandle Scene::addGaussianCloud(GaussianCloud cloud) {
    change.mark(Change::GaussianSet);
    return gaussianClouds.insert(std::move(cloud));
}

VolumeHandle Scene::addVolume(Volume volume) {
    change.mark(Change::Volumes);
    return volumes.insert(std::move(volume));
}

CameraHandle Scene::addCamera(Camera camera) {
    camera.previousTransform = camera.transform;
    const CameraHandle handle = cameras.insert(std::move(camera));
    if (!cameras.contains(activeCamera))
        activeCamera = handle;
    change.mark(Change::Camera);
    return handle;
}

NodeHandle Scene::addNode(Node node) {
    const Node* parent = nodes.find(node.parent);
    if ((node.parent && !parent) || (node.object.index() != 0 && findNode(node.object)))
        return {};
    node.worldTransform = parent ? parent->worldTransform * node.localTransform : node.localTransform;
    node.children.clear();
    const NodeObject object = node.object;
    const Core::Transform world = node.worldTransform;
    const NodeHandle parentHandle = node.parent;
    const NodeHandle handle = nodes.insert(std::move(node));
    siblingsOf(parentHandle).push_back(handle);
    if (object.index() != 0)
        objectNodes[objectKey(object)] = handle;
    placeObject(object, world);
    return handle;
}

bool Scene::remove(const MeshHandle handle) {
    if (!meshes.erase(handle))
        return false;
    change.mark(Change::MeshTopology | Change::InstanceSet);
    return true;
}

//! The subsets and instances that used the material are left without one, so their slots show empty.
bool Scene::remove(const MaterialHandle handle) {
    if (!materials.erase(handle))
        return false;
    for (Mesh& mesh : meshes.getValues())
        for (MeshSubset& subset : mesh.subsets)
            if (subset.material == handle)
                subset.material = {};
    for (Instance& instance : instances.getValues())
        if (instance.material == handle)
            instance.material = {};
    change.mark(Change::MaterialSet);
    return true;
}

bool Scene::remove(const TextureHandle handle) {
    if (!textures.erase(handle))
        return false;
    change.mark(Change::Textures);
    return true;
}

void Scene::releaseTextureData() {
    for (std::uint32_t dense = 0; dense < textures.size(); ++dense)
        if (textures.keyAt(dense) != environment.texture)
            textures.getValues()[dense].data = {};
}

bool Scene::remove(const InstanceHandle handle) {
    forgetObject(handle);
    if (!instances.erase(handle))
        return false;
    change.mark(Change::InstanceSet);
    return true;
}

bool Scene::remove(const LightHandle handle) {
    forgetObject(handle);
    if (!lights.erase(handle))
        return false;
    change.mark(Change::Lights);
    return true;
}

bool Scene::remove(const GaussianCloudHandle handle) {
    forgetObject(handle);
    if (!gaussianClouds.erase(handle))
        return false;
    change.mark(Change::GaussianSet);
    return true;
}

bool Scene::remove(const VolumeHandle handle) {
    forgetObject(handle);
    if (!volumes.erase(handle))
        return false;
    change.mark(Change::Volumes);
    return true;
}

bool Scene::remove(const CameraHandle handle) {
    forgetObject(handle);
    if (!cameras.erase(handle))
        return false;
    if (handle == activeCamera)
        activeCamera = cameras.empty() ? CameraHandle{} : cameras.keyAt(0);
    change.mark(Change::Camera);
    return true;
}

bool Scene::remove(const NodeHandle handle) {
    const Node* node = nodes.find(handle);
    if (!node)
        return false;
    std::erase(siblingsOf(node->parent), handle);
    std::vector<NodeHandle> pending{handle};
    while (!pending.empty()) {
        Node removed = std::move(*nodes.find(pending.back()));
        nodes.erase(pending.back());
        pending.pop_back();
        pending.insert(pending.end(), removed.children.begin(), removed.children.end());
        std::visit(
            [this]<class Object>(const Object& object) {
                if constexpr (!std::is_same_v<Object, std::monostate>)
                    remove(object);
            },
            removed.object);
    }
    return true;
}

bool Scene::setInstanceTransform(const InstanceHandle handle, const Core::Transform& transform) {
    Instance* instance = instances.find(handle);
    if (!instance)
        return false;
    instance->transform = transform;
    change.markInstance(instances.denseIndex(handle));
    followObject(handle, transform);
    return true;
}

bool Scene::setGaussianCloudTransform(const GaussianCloudHandle handle, const Core::Transform& transform) {
    GaussianCloud* cloud = gaussianClouds.find(handle);
    if (!cloud)
        return false;
    cloud->transform = transform;
    change.mark(Change::GaussianValues);
    followObject(handle, transform);
    return true;
}

bool Scene::setLocalTransform(const NodeHandle handle, const Core::Transform& transform) {
    Node* node = nodes.find(handle);
    if (!node)
        return false;
    const Node* parent = nodes.find(node->parent);
    node->localTransform = transform;
    node->worldTransform = parent ? parent->worldTransform * transform : transform;
    placeObject(node->object, node->worldTransform);
    placeChildren(*node);
    return true;
}

bool Scene::setParent(const NodeHandle handle, const NodeHandle parent, const std::size_t index) {
    Node* node = nodes.find(handle);
    if (!node || (parent && !nodes.contains(parent)) || isBelow(parent, handle))
        return false;
    std::erase(siblingsOf(node->parent), handle);
    std::vector<NodeHandle>& siblings = siblingsOf(parent);
    siblings.insert(siblings.begin() + static_cast<std::ptrdiff_t>(std::min(index, siblings.size())), handle);
    const Node* parentNode = nodes.find(parent);
    node->parent = parent;
    node->localTransform = parentNode ? parentNode->worldTransform.inverse() * node->worldTransform : node->worldTransform;
    return true;
}

bool Scene::setInstanceFlags(const InstanceHandle handle, const InstanceFlags flags) {
    Instance* instance = instances.find(handle);
    if (!instance)
        return false;
    instance->flags = flags;
    change.mark(Change::InstanceSet);
    return true;
}

bool Scene::setInstanceInfo(const InstanceHandle handle, const Core::float4 color, const std::uint32_t passIndex, const float random,
                            std::vector<Core::float4> attributes) {
    Instance* instance = instances.find(handle);
    if (!instance)
        return false;
    instance->color = color;
    instance->passIndex = passIndex;
    instance->random = random;
    instance->attributes = std::move(attributes);
    change.mark(Change::InstanceSet);
    return true;
}

bool Scene::setInstanceMaterial(const InstanceHandle handle, const MaterialHandle material) {
    Instance* instance = instances.find(handle);
    if (!instance)
        return false;
    instance->material = material;
    change.mark(Change::MaterialSet);
    return true;
}

bool Scene::setInstanceLightLinking(const InstanceHandle handle, const std::uint32_t lightSet, const std::uint64_t lightMask) {
    Instance* instance = instances.find(handle);
    if (!instance)
        return false;
    if (instance->lightSet == lightSet && instance->lightMask == lightMask)
        return true;
    instance->lightSet = lightSet;
    instance->lightMask = lightMask;
    change.mark(Change::InstanceSet);
    return true;
}

bool Scene::setMeshSubsetMaterial(const MeshHandle handle, const std::size_t subset, const MaterialHandle material) {
    Mesh* mesh = meshes.find(handle);
    if (!mesh)
        return false;
    if (mesh->subsets.empty() && subset == 0)
        mesh->subsets.push_back({.firstIndex = 0, .indexCount = static_cast<std::uint32_t>(mesh->indices.size())});
    if (subset >= mesh->subsets.size())
        return false;
    mesh->subsets[subset].material = material;
    change.mark(Change::MaterialSet);
    return true;
}

bool Scene::setMeshPositions(const MeshHandle handle, const std::span<const Core::float3> positions) {
    Mesh* mesh = meshes.find(handle);
    if (!mesh || positions.size() != mesh->positions.size())
        return false;
    std::ranges::copy(positions, mesh->positions.begin());
    mesh->computeBounds();
    change.markMesh(meshes.denseIndex(handle));
    return true;
}

bool Scene::setMesh(const MeshHandle handle, Mesh mesh) {
    Mesh* target = meshes.find(handle);
    if (!target)
        return false;
    const bool sameTopology = target->hasSameTopology(mesh);
    *target = std::move(mesh);
    if (sameTopology)
        change.markMesh(meshes.denseIndex(handle));
    else
        change.markMeshReplaced(meshes.denseIndex(handle));
    return true;
}

bool Scene::setMeshVertices(const MeshHandle handle, std::vector<Core::float3> positions, std::vector<std::uint32_t> normals,
                            std::vector<std::uint32_t> tangents) {
    Mesh* mesh = meshes.find(handle);
    if (!mesh || positions.size() != mesh->positions.size() || normals.size() != mesh->normals.size() || tangents.size() != mesh->tangents.size())
        return false;
    mesh->positions = std::move(positions);
    mesh->normals = std::move(normals);
    mesh->tangents = std::move(tangents);
    mesh->computeBounds();
    change.markMesh(meshes.denseIndex(handle));
    return true;
}

bool Scene::setMaterial(const MaterialHandle handle, const Material& material) {
    Material* target = materials.find(handle);
    if (!target)
        return false;
    if ((target->alphaMode == AlphaMode::Opaque) != (material.alphaMode == AlphaMode::Opaque))
        change.mark(Change::MaterialSet);
    *target = material;
    change.markMaterial(materials.denseIndex(handle));
    return true;
}

bool Scene::setLight(const LightHandle handle, const Light& light) {
    Light* target = lights.find(handle);
    if (!target)
        return false;
    Light same = light;
    //! Each sync compiles the node tree of a light again, so equal programs count as the same graph.
    if (same.graph && target->graph && same.graph->hasSameProgram(*target->graph))
        same.graph = target->graph;
    if (*target == same)
        return true;
    const bool moved = target->transform != light.transform;
    *target = light;
    change.markLight(lights.denseIndex(handle));
    if (moved)
        followObject(handle, light.transform);
    return true;
}

bool Scene::setVolume(const VolumeHandle handle, const Volume& volume) {
    Volume* target = volumes.find(handle);
    if (!target)
        return false;
    *target = volume;
    change.mark(Change::Volumes);
    return true;
}

bool Scene::setCamera(const CameraHandle handle, const Camera& camera) {
    Camera* target = cameras.find(handle);
    if (!target)
        return false;
    Camera same = camera;
    same.previousTransform = target->previousTransform;
    //! An unchanged camera keeps the accumulation; it stops moving, so its previous transform becomes the current one.
    if (same == *target) {
        target->previousTransform = target->transform;
        return true;
    }
    const Core::Transform previous = target->transform;
    *target = camera;
    target->previousTransform = previous;
    change.mark(Change::Camera);
    if (previous != camera.transform)
        followObject(handle, camera.transform);
    return true;
}

void Scene::setActiveCamera(const CameraHandle handle) {
    activeCamera = handle;
    change.mark(Change::Camera);
}

void Scene::setEnvironment(const Environment& value) {
    if (environment == value)
        return;
    environment = value;
    change.mark(Change::Environment);
}

void Scene::markGaussiansChanged() {
    change.mark(Change::GaussianValues);
}

void Scene::markVolumesChanged() {
    change.mark(Change::Volumes);
}

NodeHandle Scene::findNode(const NodeObject& object) const {
    const auto found = objectNodes.find(objectKey(object));
    return found != objectNodes.end() ? found->second : NodeHandle{};
}

Core::Bounds Scene::computeBounds() const {
    Core::Bounds bounds;
    for (const Instance& instance : instances.getValues()) {
        const Mesh* mesh = meshes.find(instance.mesh);
        if (mesh)
            bounds.grow(instance.transform.transformBounds(mesh->bounds));
    }
    for (const GaussianCloud& cloud : gaussianClouds.getValues())
        bounds.grow(cloud.transform.transformBounds(cloud.computeBounds()));
    for (const Volume& volume : volumes.getValues())
        bounds.grow(volume.transform.transformBounds(volume.bounds));
    return bounds;
}

SceneChange Scene::takeChange() {
    for (Instance& instance : instances.getValues())
        instance.previousTransform = instance.transform;
    for (Camera& camera : cameras.getValues())
        camera.previousTransform = camera.transform;
    return std::exchange(change, {});
}

void Scene::clear() {
    meshes.clear();
    materials.clear();
    textures.clear();
    instances.clear();
    lights.clear();
    gaussianClouds.clear();
    volumes.clear();
    cameras.clear();
    nodes.clear();
    rootNodes.clear();
    objectNodes.clear();
    environment = {};
    activeCamera = {};
    change.mark(lightingChanges | Change::Camera);
}

std::vector<NodeHandle>& Scene::siblingsOf(const NodeHandle parent) {
    Node* node = nodes.find(parent);
    return node ? node->children : rootNodes;
}

void Scene::placeObject(const NodeObject& object, const Core::Transform& transform) {
    if (const auto* handle = std::get_if<InstanceHandle>(&object)) {
        if (Instance* instance = instances.find(*handle)) {
            instance->transform = transform;
            change.markInstance(instances.denseIndex(*handle));
        }
    } else if (const auto* handle = std::get_if<LightHandle>(&object)) {
        if (Light* light = lights.find(*handle)) {
            light->transform = transform;
            change.markLight(lights.denseIndex(*handle));
        }
    } else if (const auto* handle = std::get_if<GaussianCloudHandle>(&object)) {
        if (GaussianCloud* cloud = gaussianClouds.find(*handle)) {
            cloud->transform = transform;
            change.mark(Change::GaussianValues);
        }
    } else if (const auto* handle = std::get_if<VolumeHandle>(&object)) {
        if (Volume* volume = volumes.find(*handle)) {
            volume->transform = transform;
            change.mark(Change::Volumes);
        }
    } else if (const auto* handle = std::get_if<CameraHandle>(&object)) {
        if (Camera* camera = cameras.find(*handle)) {
            camera->transform = transform;
            change.mark(Change::Camera);
        }
    }
}

void Scene::placeChildren(const Node& node) {
    for (const NodeHandle handle : node.children) {
        Node& child = *nodes.find(handle);
        child.worldTransform = node.worldTransform * child.localTransform;
        placeObject(child.object, child.worldTransform);
        placeChildren(child);
    }
}

void Scene::followObject(const NodeObject& object, const Core::Transform& transform) {
    Node* node = nodes.find(findNode(object));
    if (!node)
        return;
    const Node* parent = nodes.find(node->parent);
    node->localTransform = parent ? parent->worldTransform.inverse() * transform : transform;
    node->worldTransform = transform;
    placeChildren(*node);
}

void Scene::forgetObject(const NodeObject& object) {
    const auto found = objectNodes.find(objectKey(object));
    if (found == objectNodes.end())
        return;
    if (Node* node = nodes.find(found->second))
        node->object = {};
    objectNodes.erase(found);
}

bool Scene::isBelow(const NodeHandle handle, const NodeHandle ancestor) const {
    for (NodeHandle current = handle; current; current = nodes.find(current)->parent)
        if (current == ancestor)
            return true;
    return false;
}

}
