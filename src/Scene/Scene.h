#pragma once

#include <memory>
#include <vector>
#include <string>
#include <shared_mutex>
#include <atomic>
#include <mutex>

#include "Vulkan/Context.h"
#include "Vulkan/Texture.h"
#include <vulkan/vulkan.hpp>

class SceneObject;
class MeshInstance;
class MeshAsset;
class PerspectiveCamera;
class Buffer;

class Scene {
public:
    Scene(Context& context);
    ~Scene();

    // Locking functions remain the same
    std::shared_lock<std::shared_mutex> shared_lock() const {
        return std::shared_lock(sceneMutex);
    }

    std::unique_lock<std::shared_mutex> unique_lock() const {
        return std::unique_lock(sceneMutex);
    }

    // Public API for adding/removing objects remains the same
    int add(std::unique_ptr<SceneObject> sceneObject);
    void add(const std::shared_ptr<MeshAsset>& meshAsset);
    void add(Texture&& texture);
    bool remove(const SceneObject* obj);

    // Getters remain the same
    PerspectiveCamera* getActiveCamera() const { return activeCamera; }
    const std::vector<std::unique_ptr<SceneObject>>& getSceneObjects() const { return sceneObjects; }
    const std::vector<MeshInstance*>& getMeshInstances() const { return meshInstances; }
    std::shared_ptr<MeshAsset> getMeshAsset(const std::string& name) const;
    const std::vector<std::shared_ptr<MeshAsset>>& getMeshAssets() const { return meshAssets; }
    std::vector<std::string> getTextureNames() const;
    const std::vector<Texture>& getTextures() const { return textures; }
    Context& getContext() const { return context; }

    // --- SIMPLIFIED: Dirty Flag Management ---
    // Using simple atomic booleans for each flag.
    void setTlasDirty() { 
        tlasDirty.store(true, std::memory_order_relaxed);
        accumulationDirty.store(true, std::memory_order_relaxed);
    }
    
    void setMeshesDirty() { 
        meshesDirty.store(true, std::memory_order_relaxed);
        accumulationDirty.store(true, std::memory_order_relaxed);
    }
    
    void setTexturesDirty() { 
        texturesDirty.store(true, std::memory_order_relaxed);
        accumulationDirty.store(true, std::memory_order_relaxed);
    }
    
    void setAccumulationDirty() { 
        accumulationDirty.store(true, std::memory_order_relaxed);
    }

    bool isTlasDirty() const { 
        return tlasDirty.load(std::memory_order_relaxed); 
    }
    
    bool isMeshesDirty() const { 
        return meshesDirty.load(std::memory_order_relaxed); 
    }
    
    bool isTexturesDirty() const { 
        return texturesDirty.load(std::memory_order_relaxed); 
    }
    
    bool isAccumulationDirty() const { 
        return accumulationDirty.load(std::memory_order_relaxed); 
    }

    bool isAnyDirty() const {
        return tlasDirty.load(std::memory_order_relaxed) || 
               meshesDirty.load(std::memory_order_relaxed) || 
               texturesDirty.load(std::memory_order_relaxed);
    }

    void clearDirtyFlags() {
        tlasDirty.store(false, std::memory_order_relaxed);
        meshesDirty.store(false, std::memory_order_relaxed);
        texturesDirty.store(false, std::memory_order_relaxed);
        accumulationDirty.store(false, std::memory_order_relaxed);
    }

    void clearAccumulationDirtyFlag() {
        accumulationDirty.store(false, std::memory_order_relaxed);
    }

private:
    void addMeshInstance(MeshInstance* instance);

    Context& context;
    mutable std::shared_mutex sceneMutex;

    std::vector<Texture> textures;
    std::vector<std::string> textureNames;

    std::vector<std::shared_ptr<MeshAsset>> meshAssets;

    std::vector<std::unique_ptr<SceneObject>> sceneObjects;
    std::vector<MeshInstance*> meshInstances;
    PerspectiveCamera* activeCamera = nullptr;

    std::atomic<bool> tlasDirty{false};
    std::atomic<bool> meshesDirty{false};
    std::atomic<bool> texturesDirty{false};
    std::atomic<bool> accumulationDirty{false};
};
