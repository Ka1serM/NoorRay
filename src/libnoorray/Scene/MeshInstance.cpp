#include "MeshInstance.h"

#include <stdexcept>
#include <utility>

#include "Scene/Scene.h"

MeshInstance::MeshInstance(Scene& scene, const std::string& name, Mesh* mesh,
    std::vector<Material*> materials, const Transform& transf, const uint32_t materialEntry)
    : SceneObject(scene, name, transf), mesh(mesh), materials(std::move(materials)),
      materialEntry(materialEntry)
{
    for (const Material* material : this->materials)
        if (material == nullptr)
            throw std::invalid_argument("Mesh instance " + name + " has a null material");
}

MeshInstance::MeshInstance(const MeshInstance& other)
    : SceneObject(other),
      mesh(other.mesh),
      materials(other.materials),
      colors(other.colors),
      colorOwner(other.colorOwner),
      customData(other.customData),
      customDataOwner(other.customDataOwner),
      placements(other.placements),
      materialEntry(other.materialEntry),
      rayTracingFlags(other.rayTracingFlags),
      lightingChannels(other.lightingChannels)
{}

std::unique_ptr<SceneObject> MeshInstance::clone() const {
    return std::make_unique<MeshInstance>(*this);
}

void MeshInstance::markChanged() {
    if (!scene)
        return;
    scene->synchronizeBeforeMutation();
    scene->setDirtyFlag(TLAS);
    scene->setDirtyFlag(Accumulation);
    if (slot != ~0u)
        scene->markMeshInstanceChanged(slot);
}

void MeshInstance::setMaterial(const uint32_t materialSlot, Material* material) {
    if (material == nullptr)
        throw std::invalid_argument("Mesh instance " + getName() + " cannot draw with a null material");
    materials.at(materialSlot) = material;
    markChanged();
}

void MeshInstance::setMaterials(std::vector<Material*> value) {
    for (const Material* material : value)
        if (material == nullptr)
            throw std::invalid_argument("Mesh instance " + getName() + " cannot draw with a null material");
    materials = std::move(value);
    markChanged();
}

void MeshInstance::setColors(const std::span<const uint32_t> value, std::shared_ptr<const void> owner) {
    colors = value;
    colorOwner = std::move(owner);
    markChanged();
}

void MeshInstance::setPlacements(std::vector<glm::mat4> value) {
    placements = std::move(value);
    markChanged();
}

void MeshInstance::onTransformUpdated() {
    SceneObject::onTransformUpdated();
    if (!scene)
        return;
    scene->setDirtyFlag(TLAS);
    if (slot != ~0u)
        scene->markMeshInstanceChanged(slot);
}

void MeshInstance::setRayTracingFlags(const RayTracingFlags value) {
    rayTracingFlags = value;
    markChanged();
}

void MeshInstance::setCustomData(const std::span<const float> value, std::shared_ptr<const void> owner) {
    customData = value;
    customDataOwner = std::move(owner);
    markChanged();
}

void MeshInstance::setLightingChannels(const uint32_t value) {
    lightingChannels = value;
    markChanged();
}
