#include "Scene.h"
#include <ranges>
#include <iostream>
#include "Camera/PerspectiveCamera.h"
#include "Scene/MeshInstance.h"
#include "Scene/SceneObject.h"

Scene::Scene(Context& context) : context(context) {
}

Scene::~Scene() = default;

// Adds a generic SceneObject to the scene.
int Scene::add(std::unique_ptr<SceneObject> sceneObject) {
    std::unique_lock lock(sceneMutex);

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
    // Acquire a unique lock to modify the scene's data structures safely.
    std::unique_lock lock(sceneMutex);
    meshAsset->setMeshIndex(static_cast<uint32_t>(meshAssets.size()));
    meshAssets.push_back(meshAsset);
    setMeshesDirty();
}

// Adds a texture to the scene.
void Scene::add(Texture&& texture) {
    std::unique_lock lock(sceneMutex);
    textureNames.push_back(texture.getName());
    textures.push_back(std::move(texture));
    // Mark the textures as dirty.
    setTexturesDirty();
}

// Removes a specific SceneObject from the scene.
bool Scene::remove(const SceneObject* obj) {
    std::unique_lock lock(sceneMutex);

    // Prevent the removal of the active camera.
    if (activeCamera == obj) {
        std::cerr << "Warning: Cannot remove the active camera." << std::endl;
        return false;
    }

    // Check if the object is a mesh instance.
    const auto* meshInstance = dynamic_cast<const MeshInstance*>(obj);
    if (meshInstance) {
        // If it's the last mesh instance, don't allow removal.
        if ( meshInstances.size() <= 1) {
            std::cerr << "Warning: Cannot remove the last mesh instance in the scene." << std::endl;
            return false;
        }

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
    // Acquire a shared lock for read-only access.
    std::shared_lock lock(sceneMutex);
    for (const auto& meshAsset : meshAssets)
        if (meshAsset->path == name)
            return meshAsset;
    return nullptr;
}

// Retrieves the names of all loaded textures.
std::vector<std::string> Scene::getTextureNames() const {
    // Acquire a shared lock for read-only access.
    std::shared_lock lock(sceneMutex);
    return textureNames;
}
