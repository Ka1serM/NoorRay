#include "Scene.h"
#include <algorithm>
#include "Camera/CameraInstance.h"
#include "GPU/rstd/Allocator.h"
#include "Scene/LightInstance.h"
#include "Scene/MeshInstance.h"
#include "Light/PointLight.h"
#include "Light/RectLight.h"
#include "Light/SpotLight.h"
#include "Light/DirectionalLight.h"
#include "Scene/SceneObject.h"

Scene::Scene(Context& context) : context(context) {
    nr::rstd::allocator<RenderSettings> allocator;
    renderSettings = allocator.allocate(1);
    allocator.construct(renderSettings);
    renderSettings->samples = 1;
    renderSettings->diffuseBounces = 3;
    renderSettings->specularBounces = 3;
    renderSettings->transmissionBounces = 3;
    renderSettings->russianRouletteStartBounce = 3;
}

Scene::~Scene()
{
    nr::rstd::allocator<RenderSettings> allocator;
    allocator.destroy(renderSettings);
    allocator.deallocate(renderSettings, 1);

}

void Scene::notifyGeometryChanged() {
    setDirtyFlag(TLAS);
    setDirtyFlag(Accumulation);
}

// ── Internal registration ─────────────────────────────────────────────────────

uint32_t Scene::registerObject(std::unique_ptr<SceneObject> sceneObject) {
    std::shared_ptr<SceneObject> sharedObject(std::move(sceneObject));
    sharedObject->setId(nextObjectId++);

    if (auto camera = std::dynamic_pointer_cast<CameraInstance>(sharedObject))
        activeCamera = camera;

    if (auto light = std::dynamic_pointer_cast<LightInstance>(sharedObject))
        registerLight(*light);

    notifyGeometryChanged();
    sceneObjects.push_back(std::move(sharedObject));
    return static_cast<uint32_t>(sceneObjects.size() - 1);
}

// ── Public lifetime API ───────────────────────────────────────────────────────

uint64_t Scene::add(std::unique_ptr<SceneObject> sceneObject) {
    const uint32_t index = registerObject(std::move(sceneObject));
    return sceneObjects[index]->getId();
}

uint32_t Scene::add(MeshAsset meshAsset) {
    const uint32_t index = static_cast<uint32_t>(meshAssets.size());
    meshAsset.setMeshIndex(index);
    meshAssets.push_back(std::move(meshAsset));
    setDirtyFlag(Meshes);
    return index;
}

void Scene::add(Texture&& texture) {
    textureNames.push_back(texture.getName());
    textures.push_back(std::move(texture));
    setDirtyFlag(Textures);
}

bool Scene::remove(SceneObject* objToRemove) {
    if (!objToRemove || objToRemove == getActiveCamera())
        return false;

    // Recursively remove children first
    while (true) {
        const auto children = objToRemove->getChildren();
        if (children.empty()) break;
        remove(children.back().get());
    }

    if (objToRemove->getParent())
        objToRemove->getParent()->removeChild(objToRemove);

    if (auto* light = dynamic_cast<LightInstance*>(objToRemove))
        unregisterLight(*light);

    const auto it = std::ranges::find_if(sceneObjects, [objToRemove](const auto& ptr) {
        return ptr.get() == objToRemove;
    });
    if (it == sceneObjects.end())
        return false;

    if (activeObjectId == (*it)->getId())
        activeObjectId = 0;

    sceneObjects.erase(it);
    notifyGeometryChanged();
    return true;
}

bool Scene::removeObject(const uint64_t objectId) {
    return remove(findObjectPtr(objectId).get());
}

bool Scene::replaceObject(SceneObject* oldObject, std::unique_ptr<SceneObject> newObject) {
    if (!oldObject || !newObject)
        return false;

    const auto it = std::ranges::find_if(sceneObjects, [oldObject](const auto& ptr) {
        return ptr.get() == oldObject;
    });
    if (it == sceneObjects.end())
        return false;

    const uint32_t index = static_cast<uint32_t>(std::distance(sceneObjects.begin(), it));
    SceneObject* parent = oldObject->getParent();
    const auto parentPtr = findObjectPtr(parent);

    if (parent)
        parent->removeChild(oldObject);

    std::shared_ptr<SceneObject> newShared(std::move(newObject));
    newShared->setId(oldObject->getId());

    if (auto camera = std::dynamic_pointer_cast<CameraInstance>(newShared))
        activeCamera = camera;

    *it = std::move(newShared);

    if (parentPtr)
        parentPtr->addChild(sceneObjects[index]);

    notifyGeometryChanged();
    return true;
}

// ── Light management ─────────────────────────────────────────────────────────

uint32_t Scene::registerLight(LightInstance& light)
{
    uint32_t idx = UINT32_MAX;
    switch (light.lightType) {
    case LightInstance::TypePoint:
        idx = static_cast<uint32_t>(pointLights.size());
        pointLights.push_back(std::get<PointLight>(light.light));
        break;
    case LightInstance::TypeSpot:
        idx = static_cast<uint32_t>(spotLights.size());
        spotLights.push_back(std::get<SpotLight>(light.light));
        break;
    case LightInstance::TypeRect:
        idx = static_cast<uint32_t>(rectLights.size());
        rectLights.push_back(std::get<RectLight>(light.light));
        break;
    case LightInstance::TypeDirectional:
        idx = static_cast<uint32_t>(directionalLights.size());
        directionalLights.push_back(std::get<DirectionalLight>(light.light));
        break;
    }
    light.lightIndex = idx;
    setDirtyFlag(Lights);
    setDirtyFlag(Accumulation);
    return idx;
}

void Scene::unregisterLight(const LightInstance& light)
{
    const uint32_t idx = light.lightIndex;
    switch (light.lightType) {
    case LightInstance::TypePoint:
        { const uint32_t displaced = static_cast<uint32_t>(pointLights.size() - 1);
        if (idx != displaced) pointLights[idx] = std::move(pointLights.back());
        pointLights.pop_back();
        for (auto& obj : sceneObjects)
            if (auto* li = dynamic_cast<LightInstance*>(obj.get()))
                if (li->lightType == LightInstance::TypePoint && li->lightIndex == displaced)
                    { li->lightIndex = idx; break; }
        }
        break;
    case LightInstance::TypeSpot:
        { const uint32_t displaced = static_cast<uint32_t>(spotLights.size() - 1);
        if (idx != displaced) spotLights[idx] = std::move(spotLights.back());
        spotLights.pop_back();
        for (auto& obj : sceneObjects)
            if (auto* li = dynamic_cast<LightInstance*>(obj.get()))
                if (li->lightType == LightInstance::TypeSpot && li->lightIndex == displaced)
                    { li->lightIndex = idx; break; }
        }
        break;
    case LightInstance::TypeRect:
        { const uint32_t displaced = static_cast<uint32_t>(rectLights.size() - 1);
        if (idx != displaced) rectLights[idx] = std::move(rectLights.back());
        rectLights.pop_back();
        for (auto& obj : sceneObjects)
            if (auto* li = dynamic_cast<LightInstance*>(obj.get()))
                if (li->lightType == LightInstance::TypeRect && li->lightIndex == displaced)
                    { li->lightIndex = idx; break; }
        }
        break;
    case LightInstance::TypeDirectional:
        { const uint32_t displaced = static_cast<uint32_t>(directionalLights.size() - 1);
        if (idx != displaced) directionalLights[idx] = std::move(directionalLights.back());
        directionalLights.pop_back();
        for (auto& obj : sceneObjects)
            if (auto* li = dynamic_cast<LightInstance*>(obj.get()))
                if (li->lightType == LightInstance::TypeDirectional && li->lightIndex == displaced)
                    { li->lightIndex = idx; break; }
        }
        break;
    }
    setDirtyFlag(Lights);
    setDirtyFlag(Accumulation);
}

// ── Hierarchy ────────────────────────────────────────────────────────────────

void Scene::reparent(SceneObject* objectToMove, SceneObject* newParent) {
    if (!objectToMove || objectToMove == newParent)
        return;

    const auto objectToMovePtr = findObjectPtr(objectToMove);
    const auto newParentPtr = findObjectPtr(newParent);
    if (!objectToMovePtr || (newParent && !newParentPtr))
        return;

    for (SceneObject* p = newParent; p != nullptr; p = p->getParent())
        if (p == objectToMove)
            return;

    const mat4 oldWorldMatrix = objectToMove->getWorldTransform().getMatrix();

    if (objectToMove->getParent())
        objectToMove->getParent()->removeChild(objectToMove);

    if (newParentPtr) {
        newParentPtr->addChild(objectToMovePtr);
        const mat4 newLocal = inverse(newParent->getWorldTransform().getMatrix()) * oldWorldMatrix;
        objectToMove->setLocalTransform(Transform{newLocal});
    } else {
        objectToMove->clearParent();
        objectToMove->setLocalTransform(Transform{oldWorldMatrix});
    }
}

bool Scene::reparentObject(const uint64_t objectId, const uint64_t newParentId) {
    const auto objectToMove = findObjectPtr(objectId);
    const auto newParent = newParentId != 0 ? findObjectPtr(newParentId) : nullptr;
    if (!objectToMove || (newParentId != 0 && !newParent))
        return false;

    reparent(objectToMove.get(), newParent.get());
    return true;
}

// ── Clipboard ────────────────────────────────────────────────────────────────

void Scene::copyObject(const uint64_t objectId) {
    copiedObject = findObjectPtr(objectId);
}

std::shared_ptr<SceneObject> Scene::cloneHierarchy(const SceneObject* source) {
    const uint32_t index = registerObject(source->clone());
    const auto newObject = sceneObjects[index];
    for (const auto& child : source->getChildren())
        if (child)
            newObject->addChild(cloneHierarchy(child.get()));
    return newObject;
}

void Scene::paste() {
    const auto source = copiedObject.lock();
    if (!source)
        return;

    const auto newObject = cloneHierarchy(source.get());

    SceneObject* targetParent = nullptr;
    if (SceneObject* active = getActiveObject())
        targetParent = active->getParent();

    reparent(newObject.get(), targetParent);
    setActiveObjectId(newObject->getId());
}

// ── Queries ───────────────────────────────────────────────────────────────────

std::vector<std::shared_ptr<SceneObject>> Scene::getRootObjects() const {
    std::vector<std::shared_ptr<SceneObject>> result;
    for (const auto& obj : sceneObjects)
        if (obj->getParent() == nullptr)
            result.push_back(obj);
    return result;
}

std::vector<std::shared_ptr<MeshInstance>> Scene::getMeshInstances() const {
    std::vector<std::shared_ptr<MeshInstance>> result;
    for (const auto& obj : sceneObjects)
        if (auto mi = std::dynamic_pointer_cast<MeshInstance>(obj))
            result.push_back(mi);
    return result;
}

MeshAsset* Scene::getMeshAsset(const std::string& name) {
    for (auto& asset : meshAssets)
        if (asset.getPath() == name)
            return &asset;
    return nullptr;
}

const MeshAsset* Scene::getMeshAsset(const std::string& name) const {
    for (const auto& asset : meshAssets)
        if (asset.getPath() == name)
            return &asset;
    return nullptr;
}

// ── Lookups ───────────────────────────────────────────────────────────────────

std::shared_ptr<SceneObject> Scene::findObjectPtr(const SceneObject* object) const {
    if (!object) return nullptr;
    const auto it = std::ranges::find_if(sceneObjects, [object](const auto& ptr) {
        return ptr.get() == object;
    });
    return it != sceneObjects.end() ? *it : nullptr;
}

std::shared_ptr<SceneObject> Scene::findObjectPtr(const uint64_t objectId) const {
    if (objectId == 0) return nullptr;
    const auto it = std::ranges::find_if(sceneObjects, [objectId](const auto& ptr) {
        return ptr->getId() == objectId;
    });
    return it != sceneObjects.end() ? *it : nullptr;
}
