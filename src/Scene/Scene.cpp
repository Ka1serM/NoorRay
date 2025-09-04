#include "Scene.h"
#include <algorithm>
#include <iostream>
#include "Camera/PerspectiveCamera.h"
#include "Scene/MeshInstance.h"
#include "Scene/SceneObject.h"

Scene::Scene(Context& context) : context(context) {}

// Adds a generic SceneObject to the scene, making it a root object by default.
uint32_t Scene::add(std::unique_ptr<SceneObject> sceneObject) {
    if (auto* camera = dynamic_cast<PerspectiveCamera*>(sceneObject.get()))
        activeCamera = camera;
    
    if (auto* meshInstance = dynamic_cast<MeshInstance*>(sceneObject.get()))
        meshInstances.push_back(meshInstance);

    SceneObject* rawPtr = sceneObject.get();
    sceneObjects.push_back(std::move(sceneObject));
    rootObjects.push_back(rawPtr);
    
    setDirtyFlag(TLAS);
    setDirtyFlag(Accumulation);

    return sceneObjects.size() - 1;
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
    if (!objToRemove)
        return false;

    // Prevent deleting the active camera
    if (objToRemove == activeCamera) {
        reparent(objToRemove, nullptr); // Simply unparent it
        return false;
    }
    
    // Recursively remove children
    while (!objToRemove->getChildren().empty()) {
        remove(objToRemove->getChildren().back());
    }

    // --- FIX: Safely handle active object index before modification ---
    SceneObject* previouslyActiveObject = getActiveObject();
    bool activeObjectWasRemoved = (objToRemove == previouslyActiveObject);
    
    // Remove from parent or root
    if (objToRemove->getParent())
        objToRemove->getParent()->removeChild(objToRemove);
    else
        std::erase(rootObjects, objToRemove);

    // Remove from helper lists
    if (auto* meshInstance = dynamic_cast<MeshInstance*>(objToRemove))
        std::erase(meshInstances, meshInstance);

    // Erase from main ownership list
    const auto it = std::ranges::find_if(sceneObjects, [objToRemove](const auto& ptr) { return ptr.get() == objToRemove; });
    if (it != sceneObjects.end()) {
        sceneObjects.erase(it);

        // --- FIX: Recalculate the active object index ---
        if (activeObjectWasRemoved) {
            resetActiveObjectIndex(); // The active object was deleted, so deselect.
        } else if (previouslyActiveObject) {
            // Find the new index of the previously active object.
            const auto newIt = std::ranges::find_if(sceneObjects, 
                [previouslyActiveObject](const auto& ptr) { return ptr.get() == previouslyActiveObject; });
            
            if (newIt != sceneObjects.end()) {
                activeObjectIndex = std::distance(sceneObjects.begin(), newIt);
            } else {
                // This case should not happen if logic is correct
                resetActiveObjectIndex();
            }
        }
        
        setDirtyFlag(TLAS);
        setDirtyFlag(Accumulation);
        return true;
    }

    return false;
}

// Changes the parent of a SceneObject.
void Scene::reparent(SceneObject* objectToMove, SceneObject* newParent) {
    if (objectToMove == newParent)
        return; // Cannot parent to self.

    // Prevent parenting to a descendant (which would create a cycle).
    const SceneObject* p = newParent;
    while (p != nullptr) {
        if (p == objectToMove)
            return;
        p = p->getParent();
    }

    // Cache current world transform
    const Transform worldTransform = objectToMove->getWorldTransform();

    // 1. Remove from old parent
    if (objectToMove->getParent())
        objectToMove->getParent()->removeChild(objectToMove);
    else
        std::erase(rootObjects, objectToMove);

    // 2. Add to new parent
    if (newParent)
        newParent->addChild(objectToMove);
    else
        rootObjects.push_back(objectToMove);

    // 3. Update parent pointer
    objectToMove->setParent(newParent);

    // 4. Restore local transform
    if (newParent) {
        const mat4 parentWorld = newParent->getWorldTransform().getMatrix();
        const mat4 localMatrix = inverse(parentWorld) * worldTransform.getMatrix();
        objectToMove->setLocalTransform(Transform{localMatrix});
    } else
        objectToMove->setLocalTransform(worldTransform);

    objectToMove->onTransformUpdated();
}

// Retrieves a mesh asset by its name (path).
std::shared_ptr<MeshAsset> Scene::getMeshAsset(const std::string& name) const {
    for (const auto& meshAsset : meshAssets)
        if (meshAsset->getPath() == name)
            return meshAsset;
    return nullptr;
}