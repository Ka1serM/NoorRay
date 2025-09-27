#include "Scene.h"
#include <algorithm>
#include <iostream>
#include <numeric>
#include "Camera/PerspectiveCamera.h"
#include "Scene/MeshInstance.h"
#include "Scene/SceneObject.h"

Scene::Scene(Context& context) : context(context) {}

void Scene::setActiveObjectIndex(uint32_t index) {
    if (activeObjectIndex == index)
        return;

    std::cout << activeObjectIndex << std::endl;
    
    activeObjectIndex = index;
    // Notify any subscribers (like SceneGraphViewModel) that the selection changed
    Notify({SceneEvent::SelectionChanged, activeObjectIndex, dirtyFlags}); 
}

// Definition for selection getter
SceneObject* Scene::getActiveObject() const {
    if (activeObjectIndex == INVALID_INSTANCE || activeObjectIndex >= sceneObjects.size()) {
        return nullptr;
    }
    // Return the raw pointer from the unique_ptr at the stored index
    return sceneObjects[activeObjectIndex].get();
}

// Definition for dirty flag setter
void Scene::setDirtyFlag(DirtyFlag flag) {
    if ((dirtyFlags & flag) != 0)
        return; // Flag already set

    dirtyFlags |= flag;
    // Notify any subscribers that the dirty flags changed
    Notify({SceneEvent::DirtyFlagsChanged, activeObjectIndex, dirtyFlags});
}

// =========================================================================
// EXISTING FUNCTION IMPLEMENTATIONS
// =========================================================================

// Copy an object (for later paste)
void Scene::copy(SceneObject* objectToCopy) {
    copiedObject = objectToCopy;
}

// Paste the copied object into the scene
void Scene::paste() {
    if (!copiedObject) return; // Nothing to paste

    // The stable ID approach would use the object's ID, but we stick to index here.
    SceneObject* newObject = cloneHierarchy(copiedObject);

    // Paste as sibling of active object or as root
    SceneObject* targetParent = getActiveObject();
    reparent(newObject, targetParent);

    // Select the newly pasted object
    auto it = std::ranges::find_if(sceneObjects, [newObject](const auto& ptr) { return ptr.get() == newObject; });
    if (it != sceneObjects.end()) {
        uint32_t index = static_cast<uint32_t>(std::distance(sceneObjects.begin(), it));
        setActiveObjectIndex(index);
    }

    Notify({SceneEvent::HierarchyChanged});
}

// Add a new object as root
uint32_t Scene::add(std::unique_ptr<SceneObject> sceneObject) {
    SceneObject* newSceneObject = sceneObject.get();
    uint32_t index = registerObject(std::move(sceneObject));
    rootObjects.push_back(newSceneObject);
    Notify({SceneEvent::HierarchyChanged});
    return index;
}

// Register any object in the scene
uint32_t Scene::registerObject(std::unique_ptr<SceneObject> sceneObject) {
    if (auto* camera = dynamic_cast<PerspectiveCamera*>(sceneObject.get()))
        activeCamera = camera;
    else if (auto* meshInstance = dynamic_cast<MeshInstance*>(sceneObject.get())) {
        meshInstances.push_back(meshInstance);
        setDirtyFlag(TLAS);
    }

    setDirtyFlag(Accumulation);

    sceneObjects.push_back(std::move(sceneObject));
    return static_cast<uint32_t>(sceneObjects.size() - 1);
}

// Recursively clone a hierarchy
SceneObject* Scene::cloneHierarchy(const SceneObject* source) {
    std::unique_ptr<SceneObject> newObjectUPtr = source->clone();
    SceneObject* newObjectRawPtr = newObjectUPtr.get();
    registerObject(std::move(newObjectUPtr));

    for (const SceneObject* childSource : source->getChildren())
        if (childSource)
            if (SceneObject* newChild = cloneHierarchy(childSource))
                newObjectRawPtr->addChild(newChild);

    return newObjectRawPtr;
}

// Add mesh asset
void Scene::add(const std::shared_ptr<MeshAsset>& meshAsset) {
    meshAsset->setMeshIndex(static_cast<uint32_t>(meshAssets.size()));
    meshAssets.push_back(meshAsset);
    setDirtyFlag(Meshes);
}

// Add texture
void Scene::add(Texture&& texture) {
    textureNames.push_back(texture.getName());
    textures.push_back(std::move(texture));
    setDirtyFlag(Textures);
}

// Remove object from scene
bool Scene::remove(SceneObject* objToRemove) {
    if (!objToRemove || objToRemove == activeCamera) return false;

    // Recursively remove children
    while (!objToRemove->getChildren().empty())
        remove(objToRemove->getChildren().back());

    const SceneObject* previouslyActive = getActiveObject();
    bool activeRemoved = (objToRemove == previouslyActive);

    if (objToRemove->getParent())
        objToRemove->getParent()->removeChild(objToRemove);
    else
        std::erase(rootObjects, objToRemove);

    if (auto* meshInstance = dynamic_cast<MeshInstance*>(objToRemove))
        std::erase(meshInstances, meshInstance);

    auto it = std::ranges::find_if(sceneObjects, [objToRemove](const auto& ptr) { return ptr.get() == objToRemove; });
    if (it != sceneObjects.end()) {
        sceneObjects.erase(it);

        if (activeRemoved) resetActiveObjectIndex();
        else if (previouslyActive) {
            // Recalculate index of previously active object, as sceneObjects was mutated
            auto newIt = std::ranges::find_if(sceneObjects, [previouslyActive](const auto& ptr) { return ptr.get() == previouslyActive; });
            activeObjectIndex = (newIt != sceneObjects.end()) ? static_cast<uint32_t>(std::distance(sceneObjects.begin(), newIt)) : INVALID_INSTANCE;
        }

        setDirtyFlag(TLAS);
        setDirtyFlag(Accumulation);
        Notify({SceneEvent::HierarchyChanged});
        return true;
    }

    return false;
}

// Reparent object
void Scene::reparent(SceneObject* objectToMove, SceneObject* newParent) {
    if (!objectToMove || objectToMove == newParent) return;

    mat4 oldWorld = objectToMove->getWorldTransform().getMatrix();

    // Remove from current parent
    if (objectToMove->getParent())
        objectToMove->getParent()->removeChild(objectToMove);
    else
        std::erase(rootObjects, objectToMove);

    objectToMove->setParent(newParent);

    if (newParent)
        newParent->addChild(objectToMove);
    else
        rootObjects.push_back(objectToMove);

    // Update local transform
    if (newParent) {
        mat4 newLocal = inverse(newParent->getWorldTransform().getMatrix()) * oldWorld;
        objectToMove->setLocalTransform(Transform{newLocal});
    } else
        objectToMove->setLocalTransform(Transform{oldWorld});

    Notify({SceneEvent::HierarchyChanged});
}

// Get mesh asset by name
std::shared_ptr<MeshAsset> Scene::getMeshAsset(const std::string& name) const {
    for (const auto& meshAsset : meshAssets)
        if (meshAsset->getPath() == name)
            return meshAsset;
    return nullptr;
}