#include "Scene.h"
#include <algorithm>
#include <iostream>
#include "Camera/PerspectiveCamera.h"
#include "Scene/MeshInstance.h"
#include "Scene/SceneObject.h"

Scene::Scene(Context& context) : context(context) {}

// Adds a generic SceneObject to the scene, making it a root object by default.
int Scene::add(std::unique_ptr<SceneObject> sceneObject) {
    if (auto* camera = dynamic_cast<PerspectiveCamera*>(sceneObject.get()))
        activeCamera = camera;
    
    if (auto* meshInstance = dynamic_cast<MeshInstance*>(sceneObject.get()))
        meshInstances.push_back(meshInstance);

    // Get a raw pointer to the object before we move the unique_ptr
    SceneObject* rawPtr = sceneObject.get();

    // Move the object into the ownership list
    sceneObjects.push_back(std::move(sceneObject));
    
    // Add the raw pointer to the root object list for the hierarchy
    rootObjects.push_back(rawPtr);
    
    setDirtyFlag(TLAS);
    setDirtyFlag(Accumulation);

    return sceneObjects.size() - 1; // Return the index of the newly added object
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

bool Scene::remove(SceneObject* objToRemove) {
    if (!objToRemove)
        return false;

    // If it's the active camera, just reparent it to root instead of deleting
    if (objToRemove == activeCamera) {
        if (objToRemove->getParent())
            objToRemove->getParent()->removeChild(objToRemove);
        rootObjects.push_back(objToRemove);
        return false; // Do not delete camera
    }

    // Recursively remove children, but skip active camera
    while (!objToRemove->getChildren().empty()) {
        if (SceneObject* child = objToRemove->getChildren().back(); child == activeCamera) {
            // Reparent camera to root instead of deleting
            objToRemove->removeChild(child);
            rootObjects.push_back(child);
        } else
            remove(child); // Safe recursive delete
    }

    // Remove from parent's children list OR from the root list
    if (objToRemove->getParent())
        objToRemove->getParent()->removeChild(objToRemove);
    else
        std::erase(rootObjects, objToRemove);

    // Reset active object index if necessary
    if (objToRemove == getActiveObject())
        setActiveObjectIndex(-1);

    if (auto* meshInstance = dynamic_cast<MeshInstance*>(objToRemove))
        std::erase(meshInstances, meshInstance);

    // Erase the object from main ownership list
    if (const auto it = std::ranges::find_if(sceneObjects, [objToRemove](const auto& ptr) { return ptr.get() == objToRemove; });
        it != sceneObjects.end()) 
    {
        sceneObjects.erase(it);
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