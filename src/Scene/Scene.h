#pragma once

#include <memory>
#include <vector>
#include <string>
#include "Vulkan/Context.h"
#include "Vulkan/Texture.h"
#include <vulkan/vulkan.hpp>

class SceneObject;
class MeshInstance;
class MeshAsset;
class PerspectiveCamera;
class Buffer;

// Dirty flags as bitfield
enum DirtyFlag : uint8_t {
    TLAS         = 1 << 0,
    Meshes       = 1 << 1,
    Textures     = 1 << 2,
    Accumulation = 1 << 3,
};

class Scene {
public:
    Scene(Context& context);
    
    // Object management
    int add(std::unique_ptr<SceneObject> sceneObject);
    void add(const std::shared_ptr<MeshAsset>& meshAsset);
    void add(Texture&& texture);
    bool remove(SceneObject* objToRemove);
    void reparent(SceneObject* objectToMove, SceneObject* newParent);

    // Getters for scene content
    PerspectiveCamera* getActiveCamera() const { return activeCamera; }
    const std::vector<std::unique_ptr<SceneObject>>& getSceneObjects() const { return sceneObjects; }
    const std::vector<SceneObject*>& getRootObjects() const { return rootObjects; }

    SceneObject* getObject(const int index) const {
        if (index >= 0 && index < static_cast<int>(sceneObjects.size()))
            return sceneObjects[index].get();
        return nullptr;
    }
    
    // Index-based selection
    void setActiveObjectIndex(const uint32_t index) { activeObjectIndex = index; }
    uint32_t getActiveObjectIndex() const { return activeObjectIndex; }
    SceneObject* getActiveObject() const {
        if (activeObjectIndex >= 0 && activeObjectIndex < sceneObjects.size())
            return sceneObjects[activeObjectIndex].get();
        return nullptr;
    }
    
    const std::vector<MeshInstance*>& getMeshInstances() const { return meshInstances; }
    std::shared_ptr<MeshAsset> getMeshAsset(const std::string& name) const;
    const std::vector<std::shared_ptr<MeshAsset>>& getMeshAssets() const { return meshAssets; }
    const std::vector<Texture>& getTextures() const { return textures; }
    Context& getContext() const { return context; }
    std::vector<std::string> getTextureNames() const { return textureNames; }

    // Dirty flag management
    void setDirtyFlag(const DirtyFlag flag) { dirtyFlags |= flag; }
    void clearDirtyFlag(const DirtyFlag flag) { dirtyFlags &= ~flag; }
    bool isDirty(const DirtyFlag flag) const { return (dirtyFlags & flag) != 0; }
    bool isAnyDirty() const { return dirtyFlags & (TLAS | Meshes | Textures); }
    void clearDirtyFlags() { dirtyFlags = 0; }
    void clearAccumulationDirtyFlag() { dirtyFlags &= ~Accumulation; }

private:
    Context& context;

    std::vector<Texture> textures;
    std::vector<std::string> textureNames;
    std::vector<std::shared_ptr<MeshAsset>> meshAssets;

    // Owns ALL objects in the scene.
    std::vector<std::unique_ptr<SceneObject>> sceneObjects;
    // Non-owning pointers to top-level objects for the hierarchy.
    std::vector<SceneObject*> rootObjects;

    std::vector<MeshInstance*> meshInstances;
    PerspectiveCamera* activeCamera = nullptr;
    uint32_t activeObjectIndex = -1;
    uint8_t dirtyFlags = 0;
};