#include "Scene.h"
#include <ranges>
#include <iostream>
#include "Camera/PerspectiveCamera.h"
#include "Scene/MeshInstance.h"
#include "Scene/SceneObject.h"

Scene::Scene(Context& context) : context(context) {
}

// Adds a generic SceneObject to the scene.
int Scene::add(std::unique_ptr<SceneObject> sceneObject) {
    if (auto* camera = dynamic_cast<PerspectiveCamera*>(sceneObject.get()))
        activeCamera = camera;
    
    if (auto* meshInstance = dynamic_cast<MeshInstance*>(sceneObject.get()))
        addMeshInstance(meshInstance);

    // Add the object to the main list and mark the TLAS as dirty.
    sceneObjects.push_back(std::move(sceneObject));
    setTlasDirty();
    return static_cast<int>(sceneObjects.size() - 1);
}

// Adds a mesh asset to the scene.
void Scene::add(const std::shared_ptr<MeshAsset>& meshAsset) {
    meshAsset->setMeshIndex(static_cast<uint32_t>(meshAssets.size()));
    meshAssets.push_back(meshAsset);
    setMeshesDirty();
}

// Adds a texture to the scene.
void Scene::add(Texture&& texture) {
    textureNames.push_back(texture.getName());
    textures.push_back(std::move(texture));
    // Mark the textures as dirty.
    setTexturesDirty();
}

// Removes a specific SceneObject from the scene.
bool Scene::remove(const SceneObject* obj) {
    // Prevent the removal of the active camera.
    if (activeCamera == obj) {
        std::cerr << "Warning: Cannot remove the active camera." << std::endl;
        return false;
    }

    // Check if the object is a mesh instance.
    const auto* meshInstance = dynamic_cast<const MeshInstance*>(obj);
    if (meshInstance) {
        const auto it = std::ranges::find(meshInstances, meshInstance);
        if (it != meshInstances.end())
            meshInstances.erase(it);
    }

    const auto it = std::ranges::find_if(sceneObjects,
        [obj](const std::unique_ptr<SceneObject>& ptr) {
            return ptr.get() == obj;
        });

    // If found, erase it and mark the TLAS as dirty.
    if (it != sceneObjects.end()) {
        sceneObjects.erase(it);
        setTlasDirty();
        return true;
    }

    return false;
}

// Private helper to add a mesh instance. No lock needed as it's called from a locked context.
void Scene::addMeshInstance(MeshInstance* instance) {
    meshInstances.push_back(instance);
}

// Retrieves a mesh asset by its name (path).
std::shared_ptr<MeshAsset> Scene::getMeshAsset(const std::string& name) const {
    for (const auto& meshAsset : meshAssets)
        if (meshAsset->getPath() == name)
            return meshAsset;
    return nullptr;
}

// Retrieves the names of all loaded textures.
std::vector<std::string> Scene::getTextureNames() const {
    return textureNames;
}
