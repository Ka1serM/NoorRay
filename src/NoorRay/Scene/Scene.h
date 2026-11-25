#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

#include "NoorRay/Core/SlotMap.h"
#include "NoorRay/Scene/Camera.h"
#include "NoorRay/Scene/Environment.h"
#include "NoorRay/Scene/GaussianCloud.h"
#include "NoorRay/Scene/Instance.h"
#include "NoorRay/Scene/Light.h"
#include "NoorRay/Scene/Material.h"
#include "NoorRay/Scene/Mesh.h"
#include "NoorRay/Scene/Node.h"
#include "NoorRay/Scene/SceneChange.h"
#include "NoorRay/Scene/Texture.h"
#include "NoorRay/Scene/Volume.h"

namespace NoorRay::Scene {

using MeshHandle = Core::Handle<Mesh>;
using MaterialHandle = Core::Handle<Material>;
using TextureHandle = Core::Handle<Texture>;
using InstanceHandle = Core::Handle<Instance>;
using LightHandle = Core::Handle<Light>;
using GaussianCloudHandle = Core::Handle<GaussianCloud>;
using VolumeHandle = Core::Handle<Volume>;
using CameraHandle = Core::Handle<Camera>;
using NodeHandle = Core::Handle<Node>;

//! The host scene. Every edit goes through a method, so that the next GpuScene update knows exactly what changed.
class Scene {
public:
    MeshHandle addMesh(Mesh mesh);
    MaterialHandle addMaterial(Material material);
    TextureHandle addTexture(Texture texture);
    InstanceHandle addInstance(Instance instance);
    LightHandle addLight(Light light);
    GaussianCloudHandle addGaussianCloud(GaussianCloud cloud);
    VolumeHandle addVolume(Volume volume);
    CameraHandle addCamera(Camera camera);
    //! Appends the node to the children of its parent, or to the roots, and moves its object to the world transform of the node.
    NodeHandle addNode(Node node);

    bool remove(MeshHandle handle);
    bool remove(MaterialHandle handle);
    bool remove(TextureHandle handle);
    //! Frees the texels of every texture but the environment map, which the GPU scene reads again; only after the GPU scene took them.
    void releaseTextureData();
    bool remove(InstanceHandle handle);
    bool remove(LightHandle handle);
    bool remove(GaussianCloudHandle handle);
    bool remove(VolumeHandle handle);
    bool remove(CameraHandle handle);
    //! Removes the node, its children and their objects.
    bool remove(NodeHandle handle);

    //! An object transform is a world transform; the node of the object follows it.
    bool setInstanceTransform(InstanceHandle handle, const Core::Transform& transform);
    bool setGaussianCloudTransform(GaussianCloudHandle handle, const Core::Transform& transform);
    bool setLocalTransform(NodeHandle handle, const Core::Transform& transform);
    //! Moves the node to position index among the children of parent, or of the roots, and keeps its world transform. A node never moves below itself.
    bool setParent(NodeHandle handle, NodeHandle parent, std::size_t index);
    bool setInstanceFlags(InstanceHandle handle, InstanceFlags flags);
    bool setInstanceInfo(InstanceHandle handle, Core::float4 color, std::uint32_t passIndex, float random, std::vector<Core::float4> attributes);
    bool setInstanceMaterial(InstanceHandle handle, MaterialHandle material);
    bool setInstanceLightLinking(InstanceHandle handle, std::uint32_t lightSet, std::uint64_t lightMask);
    //! Sets the material of one subset; subset 0 of a mesh without subsets becomes one subset over the whole mesh.
    bool setMeshSubsetMaterial(MeshHandle handle, std::size_t subset, MaterialHandle material);
    bool setMeshPositions(MeshHandle handle, std::span<const Core::float3> positions);
    //! Replaces the mesh data; a mesh with the same topology only updates its vertex streams.
    bool setMesh(MeshHandle handle, Mesh mesh);
    //! Replaces the positions, normals and tangents of a mesh; each stream keeps its length, or is empty where the mesh has none.
    bool setMeshVertices(MeshHandle handle, std::vector<Core::float3> positions, std::vector<std::uint32_t> normals, std::vector<std::uint32_t> tangents);
    //! Lets write(mesh) change the streams of a mesh in place, at the same sizes; with indices, the mesh builds its acceleration structure again.
    template <class Write>
    bool updateMesh(const MeshHandle handle, const bool indices, Write&& write) {
        Mesh* mesh = meshes.find(handle);
        if (!mesh)
            return false;
        const std::array sizes{mesh->positions.size(), mesh->indices.size(), mesh->normals.size(), mesh->tangents.size(), mesh->colors.size()};
        if (!write(*mesh))
            return false;
        if (sizes != std::array{mesh->positions.size(), mesh->indices.size(), mesh->normals.size(), mesh->tangents.size(), mesh->colors.size()})
            return false;
        mesh->computeBounds();
        if (indices)
            change.markMeshReplaced(meshes.denseIndex(handle));
        else
            change.markMesh(meshes.denseIndex(handle));
        return true;
    }
    bool setMaterial(MaterialHandle handle, const Material& material);
    bool setLight(LightHandle handle, const Light& light);
    bool setVolume(VolumeHandle handle, const Volume& volume);
    bool setCamera(CameraHandle handle, const Camera& camera);
    void setActiveCamera(CameraHandle handle);
    void setEnvironment(const Environment& environment);
    void markGaussiansChanged();
    void markVolumesChanged();

    const Mesh* find(MeshHandle handle) const { return meshes.find(handle); }
    const Material* find(MaterialHandle handle) const { return materials.find(handle); }
    const Texture* find(TextureHandle handle) const { return textures.find(handle); }
    const Instance* find(InstanceHandle handle) const { return instances.find(handle); }
    const Light* find(LightHandle handle) const { return lights.find(handle); }
    const GaussianCloud* find(GaussianCloudHandle handle) const { return gaussianClouds.find(handle); }
    const Volume* find(VolumeHandle handle) const { return volumes.find(handle); }
    const Camera* find(CameraHandle handle) const { return cameras.find(handle); }
    const Node* find(NodeHandle handle) const { return nodes.find(handle); }
    NodeHandle findNode(const NodeObject& object) const;
    GaussianCloud* findForTraining(GaussianCloudHandle handle) { return gaussianClouds.find(handle); }

    const Core::SlotMap<Mesh>& getMeshes() const { return meshes; }
    const Core::SlotMap<Material>& getMaterials() const { return materials; }
    const Core::SlotMap<Texture>& getTextures() const { return textures; }
    const Core::SlotMap<Instance>& getInstances() const { return instances; }
    const Core::SlotMap<Light>& getLights() const { return lights; }
    const Core::SlotMap<GaussianCloud>& getGaussianClouds() const { return gaussianClouds; }
    const Core::SlotMap<Volume>& getVolumes() const { return volumes; }
    const Core::SlotMap<Camera>& getCameras() const { return cameras; }
    const Core::SlotMap<Node>& getNodes() const { return nodes; }
    std::span<const NodeHandle> getRootNodes() const { return rootNodes; }
    const Environment& getEnvironment() const { return environment; }
    CameraHandle getActiveCamera() const { return activeCamera; }
    const Camera* findActiveCamera() const { return cameras.find(activeCamera); }

    Core::Bounds computeBounds() const;
    const SceneChange& getChange() const { return change; }
    SceneChange takeChange();
    void clear();

private:
    std::vector<NodeHandle>& siblingsOf(NodeHandle parent);
    void placeObject(const NodeObject& object, const Core::Transform& transform);
    void placeChildren(const Node& node);
    void followObject(const NodeObject& object, const Core::Transform& transform);
    void forgetObject(const NodeObject& object);
    bool isBelow(NodeHandle handle, NodeHandle ancestor) const;

    Core::SlotMap<Mesh> meshes;
    Core::SlotMap<Material> materials;
    Core::SlotMap<Texture> textures;
    Core::SlotMap<Instance> instances;
    Core::SlotMap<Light> lights;
    Core::SlotMap<GaussianCloud> gaussianClouds;
    Core::SlotMap<Volume> volumes;
    Core::SlotMap<Camera> cameras;
    Core::SlotMap<Node> nodes;
    std::vector<NodeHandle> rootNodes;
    std::unordered_map<std::uint64_t, NodeHandle> objectNodes;
    Environment environment;
    CameraHandle activeCamera;
    SceneChange change;
};

}
