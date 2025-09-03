#include "Scene.h"
#include <algorithm>
#include <iostream>
#include "Camera/PerspectiveCamera.h"
#include "Scene/MeshInstance.h"
#include "Scene/SceneObject.h"

Scene::Scene(Context& context) : context(context) {}

void Scene::copy(SceneObject* objectToCopy)
{
    copiedObject = objectToCopy;
}

void Scene::paste() {
    if (!copiedObject)
        return; // Nothing to paste

    //Create a  clone of the copied object and its entire hierarchy.
    SceneObject* newObject = cloneHierarchy(copiedObject);

    //    - If an object is selected, paste as a sibling.
    //    - Otherwise, paste as a root object.
    SceneObject* targetParent = nullptr;
    if (SceneObject* activeObject = getActiveObject())
        targetParent = activeObject;

    // Reparent the new hierarchy to its correct place in the scene.
    reparent(newObject, targetParent);

    // Select the top-level object of the newly pasted hierarchy.
    const auto it = std::ranges::find_if(sceneObjects,  [newObject](const auto& ptr) { return ptr.get() == newObject; });
    if (it != sceneObjects.end()) {
        const uint32_t index = static_cast<uint32_t>(std::distance(sceneObjects.begin(), it));
        setActiveObjectIndex(index);
    }
}

uint32_t Scene::add(std::unique_ptr<SceneObject> sceneObject) {
    SceneObject* newSceneObject = sceneObject.get();
    
    // It calls the private helper to handle the actual registration.
    const uint32_t addedIndex = registerObject(std::move(sceneObject));
    
    // Then it completes its job by adding the object to the root.
    rootObjects.push_back(newSceneObject);
    
    return addedIndex;
}

uint32_t Scene::registerObject(std::unique_ptr<SceneObject> sceneObject) {
    if (auto* camera = dynamic_cast<PerspectiveCamera*>(sceneObject.get()))
        activeCamera = camera;
    else if (auto* meshInstance = dynamic_cast<MeshInstance*>(sceneObject.get()))
    {
        meshInstances.push_back(meshInstance);
        setDirtyFlag(TLAS);
    }
    setDirtyFlag(Accumulation);

    sceneObjects.push_back(std::move(sceneObject));
    return static_cast<uint32_t>(sceneObjects.size() - 1);
}

SceneObject* Scene::cloneHierarchy(const SceneObject* source) {
    std::unique_ptr<SceneObject> newObjectUPtr = source->clone();
    SceneObject* newObjectRawPtr = newObjectUPtr.get();

    registerObject(std::move(newObjectUPtr));
    for (const SceneObject* childSource : source->getChildren())
        if (childSource != nullptr)
            if (SceneObject* newChild = cloneHierarchy(childSource))
                newObjectRawPtr->addChild(newChild);
    
    return newObjectRawPtr;
}

// Adds a mesh asset to the scene.
void Scene::add(const std::shared_ptr<MeshAsset>& meshAsset) {
    meshAsset->setMeshIndex(static_cast<uint32_t>(meshAssets.size()));
    meshAssets.push_back(meshAsset);
    setDirtyFlag(Meshes);
}

// Adds a texture to the scene.
void Scene::add(Texture&& texture) {
    textureNames.push_back(texture.getName());
    textures.push_back(std::move(texture));
    setDirtyFlag(Textures);
}

// In Scene.cpp
bool Scene::remove(SceneObject* objToRemove) {
    if (!objToRemove || objToRemove == activeCamera)
        return false;;
    
    // Recursively remove children
    while (!objToRemove->getChildren().empty())
        remove(objToRemove->getChildren().back());

    const SceneObject* previouslyActiveObject = getActiveObject();
    const bool activeObjectWasRemoved = (objToRemove == previouslyActiveObject);
    
    if (objToRemove->getParent())
        objToRemove->getParent()->removeChild(objToRemove);
    else
        std::erase(rootObjects, objToRemove);

    if (auto* meshInstance = dynamic_cast<MeshInstance*>(objToRemove))
        std::erase(meshInstances, meshInstance);

    const auto it = std::ranges::find_if(sceneObjects, [objToRemove](const auto& ptr) { return ptr.get() == objToRemove; });
    if (it != sceneObjects.end()) {
        const uint32_t removedIndex = static_cast<uint32_t>(std::distance(sceneObjects.begin(), it));
        sceneObjects.erase(it);

        if (activeObjectWasRemoved) {
            resetActiveObjectIndex();
        } else if (previouslyActiveObject) {
            const auto newIt = std::ranges::find_if(sceneObjects, 
                [previouslyActiveObject](const auto& ptr) { return ptr.get() == previouslyActiveObject; });
            
            if (newIt != sceneObjects.end()) {
                activeObjectIndex = static_cast<uint32_t>(std::distance(sceneObjects.begin(), newIt));
            } else
                resetActiveObjectIndex();
        }
        
        setDirtyFlag(TLAS);
        setDirtyFlag(Accumulation);
        return true;
    }

    return false;
}

void Scene::reparent(SceneObject* objectToMove, SceneObject* newParent) {
    if (!objectToMove || objectToMove == newParent)
        return;

    // 1. Get the current world transform *before* modifying the parent.
    const mat4 oldWorldMatrix = objectToMove->getWorldTransform().getMatrix();

    // 2. Unparent the object from its current parent.
    if (objectToMove->getParent())
        objectToMove->getParent()->removeChild(objectToMove);
    else
        std::erase(rootObjects, objectToMove);

    // 3. Set the new parent.
    objectToMove->setParent(newParent);

    // 4. Add the object to its new parent's child list or to the root.
    if (newParent)
        newParent->addChild(objectToMove);
    else
        rootObjects.push_back(objectToMove);

    // 5. Calculate and set the new local transform based on the old world transform.
    if (newParent) {
        const mat4 newParentWorldMatrix = newParent->getWorldTransform().getMatrix();
        const mat4 newLocalMatrix = inverse(newParentWorldMatrix) * oldWorldMatrix;
        objectToMove->setLocalTransform(Transform{newLocalMatrix});
    } else
        objectToMove->setLocalTransform(Transform{oldWorldMatrix});
}

std::shared_ptr<MeshAsset> Scene::getMeshAsset(const std::string& name) const {
    for (const auto& meshAsset : meshAssets)
        if (meshAsset->getPath() == name)
            return meshAsset;
    return nullptr;
}