#include "Scene.h"
#include <algorithm>
#include "Camera/CameraBase.h"
#include "Scene/MeshInstance.h"
#include "Scene/SceneObject.h"

Scene::Scene(Context& context) : context(context) {
    environment.settings.textureIndex = 0;
    environment.settings.cdfTextureIndex = 1;
}

void Scene::notifyGeometryChanged() {
    setDirtyFlag(TLAS);
    setDirtyFlag(Accumulation);
    setDirtyFlag(RayLut);
}

// ── Internal registration ─────────────────────────────────────────────────────

uint32_t Scene::registerObject(std::unique_ptr<SceneObject> sceneObject) {
    std::shared_ptr<SceneObject> sharedObject(std::move(sceneObject));
    sharedObject->setId(nextObjectId++);

    if (auto camera = std::dynamic_pointer_cast<CameraBase>(sharedObject))
        activeCamera = camera;

    notifyGeometryChanged();
    sceneObjects.push_back(std::move(sharedObject));
    return static_cast<uint32_t>(sceneObjects.size() - 1);
}

// ── Public lifetime API ───────────────────────────────────────────────────────

uint64_t Scene::add(std::unique_ptr<SceneObject> sceneObject) {
    const uint32_t index = registerObject(std::move(sceneObject));
    return sceneObjects[index]->getId();
}

void Scene::add(const std::shared_ptr<MeshAsset>& meshAsset) {
    meshAsset->setMeshIndex(static_cast<uint32_t>(meshAssets.size()));
    meshAssets.push_back(meshAsset);
    setDirtyFlag(Meshes);
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

    if (auto camera = std::dynamic_pointer_cast<CameraBase>(newShared))
        activeCamera = camera;

    *it = std::move(newShared);

    if (parentPtr)
        parentPtr->addChild(sceneObjects[index]);

    notifyGeometryChanged();
    return true;
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

std::shared_ptr<MeshAsset> Scene::getMeshAsset(const std::string& name) const {
    for (const auto& asset : meshAssets)
        if (asset->getPath() == name)
            return asset;
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
