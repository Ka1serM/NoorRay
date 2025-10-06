#include "Scene.h"
#include <algorithm>
#include <iostream>
#include "Camera/PerspectiveCamera.h"
#include "Scene/MeshInstance.h"
#include "Scene/SceneObject.h"

Scene::Scene(Context& context) : context(context) {}

// -----------------------
// Object Management
// -----------------------

// Default add, places object at the root of the scene
int Scene::add(std::unique_ptr<SceneObject> sceneObject) {
    return add(std::move(sceneObject), -1); // Default to no parent
}

// Add with explicit parent, for direct hierarchy construction
int Scene::add(std::unique_ptr<SceneObject> sceneObject, int parentId) {
    if (!sceneObject) return -1;

    SceneObject* rawPtr = sceneObject.get();
    const int id = nextObjectId++;
    rawPtr->setId(id);

    // Store ownership in the map
    objectsById[id] = std::move(sceneObject);

    // Get parent
    SceneObject* parent = getObject(parentId);

    if (parent) {
        rawPtr->setParent(parent);
        parent->addChild(rawPtr);
    } else {
        // Add to root if no (valid) parent
        rootObjects.push_back(rawPtr);
    }
    
    // Handle special types
    if (auto* camera = dynamic_cast<PerspectiveCamera*>(rawPtr))
        activeCamera = camera;
    else if (auto* meshInstance = dynamic_cast<MeshInstance*>(rawPtr))
        meshInstances.push_back(meshInstance);

    setDirtyFlag(Accumulation);
    if (dynamic_cast<MeshInstance*>(rawPtr))
        setDirtyFlag(TLAS);

    Notify({SceneEvent::HierarchyChanged});
    return id;
}


bool Scene::remove(SceneObject* objToRemove) {
    if (!objToRemove || objToRemove == activeCamera) return false;

    // Remove children recursively
    auto children = objToRemove->getChildren(); 
    for (auto* child : children)
        remove(child);

    // Remove from parent or root
    if (auto* parent = objToRemove->getParent())
        parent->removeChild(objToRemove);
    else
        std::erase(rootObjects, objToRemove);

    // Remove from meshInstances if applicable
    if (auto* meshInstance = dynamic_cast<MeshInstance*>(objToRemove))
        std::erase(meshInstances, meshInstance);

    // Remove from objectsById using the object's ID
    const int idToRemove = objToRemove->getId();
    if (objectsById.count(idToRemove)) {
        objectsById.erase(idToRemove);

        // Update active object if removed
        if (activeObjectId == idToRemove)
            activeObjectId = -1;

        setDirtyFlag(TLAS);
        setDirtyFlag(Accumulation);
        Notify({SceneEvent::HierarchyChanged});
        return true;
    }

    return false;
}

void Scene::reparent(SceneObject* objectToMove, SceneObject* newParent) {
    if (!objectToMove || objectToMove == newParent || objectToMove->getParent() == newParent) return;

    mat4 oldWorld = objectToMove->getWorldTransform().getMatrix();

    // Remove from old parent or root
    if (auto* oldParent = objectToMove->getParent())
        oldParent->removeChild(objectToMove);
    else
        std::erase(rootObjects, objectToMove);

    // Set new parent
    objectToMove->setParent(newParent);
    if (newParent)
        newParent->addChild(objectToMove);
    else
        rootObjects.push_back(objectToMove);

    // Update local transform to preserve world position
    if (newParent) {
        mat4 newLocal = inverse(newParent->getWorldTransform().getMatrix()) * oldWorld;
        objectToMove->setLocalTransform(Transform{newLocal});
    } else
        objectToMove->setLocalTransform(Transform{oldWorld});

    Notify({SceneEvent::HierarchyChanged});
}

// -----------------------
// Copy / Paste
// -----------------------

void Scene::copy(SceneObject* objectToCopy) {
    copiedObject = objectToCopy;
}

SceneObject* Scene::cloneHierarchy(const SceneObject* source) {
    if (!source) return nullptr;

    std::unique_ptr<SceneObject> newObjUPtr = source->clone();
    SceneObject* newObj = newObjUPtr.get();
    
    // Add the cloned object to the scene to get a valid ID.
    // It will be a root object temporarily.
    const int newId = add(std::move(newObjUPtr));

    for (const auto* childSource : source->getChildren()) {
        if (SceneObject* newChild = cloneHierarchy(childSource)) {
             // Reparent the new child under the new object.
             // This correctly sets up the hierarchy for the cloned objects.
            reparent(newChild, newObj);
        }
    }

    return newObj;
}


void Scene::paste() {
    if (!copiedObject) return;

    SceneObject* newObj = cloneHierarchy(copiedObject);
    if (!newObj) return;

    // Reparent under active object or root
    SceneObject* targetParent = getActiveObject();
    reparent(newObj, targetParent);

    // Select newly pasted object
    setActiveObject(newObj->getId());

    Notify({SceneEvent::HierarchyChanged});
}

// -----------------------
// Active Object
// -----------------------

void Scene::setActiveObject(int id) {
    if (activeObjectId == id) return;
    
    // Allow setting to -1 (no selection)
    if (id != -1 && !getObject(id)) return;

    activeObjectId = id;
    Notify({SceneEvent::SelectionChanged, static_cast<uint32_t>(id), dirtyFlags});
}

// -----------------------
// Assets
// -----------------------

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

std::shared_ptr<MeshAsset> Scene::getMeshAsset(const std::string& name) const {
    for (const auto& asset : meshAssets)
        if (asset->getPath() == name)
            return asset;
    return nullptr;
}

// -----------------------
// Dirty Flags
// -----------------------

void Scene::setDirtyFlag(DirtyFlag flag) {
    if ((dirtyFlags & flag) != 0) return;
    dirtyFlags |= flag;
    Notify({SceneEvent::DirtyFlagsChanged, static_cast<uint32_t>(activeObjectId), dirtyFlags});
}

