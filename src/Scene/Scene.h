#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include <string>
#include "Vulkan/Context.h"
#include "Vulkan/Texture.h"
#include <vulkan/vulkan.hpp>

#include "Scene/SceneTypes.h"
#include "Scene/Environment.h"

class SceneObject;
class MeshInstance;
class MeshAsset;
class CameraInstance;
class Buffer;

enum DirtyFlag : uint8_t {
    TLAS         = 1 << 0,
    Meshes       = 1 << 1,
    Textures     = 1 << 2,
    Accumulation = 1 << 3,
};

struct RenderSettings
{
    int samples{};
    int diffuseBounces{};
    int specularBounces{};
    int transmissionBounces{};
    int adaptiveSamplingEnabled{};
    int adaptiveMinSamples{};
    float adaptiveTargetError{};
    int russianRouletteStartBounce{};
    float exposure{};
    int transparentBackground{};
    int renderMode{};
    int bufferVisualization{};
    int taaEnabled{};
    int aoSampleAlbedo{};
};

class Scene {
    Context& context;

    std::vector<Texture> textures;
    std::vector<std::string> textureNames;
    std::vector<std::shared_ptr<MeshAsset>> meshAssets;
    Environment environment{};
    RenderSettings* renderSettings{};

    std::vector<std::shared_ptr<SceneObject>> sceneObjects;

    std::weak_ptr<CameraInstance> activeCamera;
    uint64_t activeObjectId = 0;
    uint64_t nextObjectId = 1;
    uint8_t dirtyFlags = 0;

    std::weak_ptr<SceneObject> copiedObject;

    std::shared_ptr<SceneObject> findObjectPtr(const SceneObject* object) const;
    std::shared_ptr<SceneObject> findObjectPtr(uint64_t objectId) const;

    uint32_t registerObject(std::unique_ptr<SceneObject> sceneObject);
    bool remove(SceneObject* objToRemove);
    void reparent(SceneObject* objectToMove, SceneObject* newParent);
    std::shared_ptr<SceneObject> cloneHierarchy(const SceneObject* source);
    void notifyGeometryChanged();

public:
    Scene(Context& context);
    ~Scene();

    // Object lifetime
    uint64_t add(std::unique_ptr<SceneObject> sceneObject);
    void add(const std::shared_ptr<MeshAsset>& meshAsset);
    void add(Texture&& texture);
    bool removeObject(uint64_t objectId);
    bool replaceObject(SceneObject* oldObject, std::unique_ptr<SceneObject> newObject);

    // Hierarchy
    bool reparentObject(uint64_t objectId, uint64_t newParentId = 0);

    // Clipboard
    void copyObject(uint64_t objectId);
    void paste();

    // Lookup
    SceneObject* getObject(uint64_t objectId) const { return findObjectPtr(objectId).get(); }
    std::shared_ptr<SceneObject> getObjectPtr(uint64_t objectId) const { return findObjectPtr(objectId); }
    const std::vector<std::shared_ptr<SceneObject>>& getSceneObjects() const { return sceneObjects; }
    std::vector<std::shared_ptr<SceneObject>> getRootObjects() const;
    std::vector<std::shared_ptr<MeshInstance>> getMeshInstances() const;
    std::shared_ptr<MeshAsset> getMeshAsset(const std::string& name) const;
    const std::vector<std::shared_ptr<MeshAsset>>& getMeshAssets() const { return meshAssets; }
    const std::vector<Texture>& getTextures() const { return textures; }
    std::vector<std::string> getTextureNames() const { return textureNames; }

    // Active object
    void setActiveObjectId(uint64_t objectId) { activeObjectId = objectId; }
    void clearActiveObject() { activeObjectId = 0; }
    uint64_t getActiveObjectId() const { return activeObjectId; }
    SceneObject* getActiveObject() const { return findObjectPtr(activeObjectId).get(); }
    std::shared_ptr<SceneObject> getActiveObjectPtr() const { return findObjectPtr(activeObjectId); }

    // Camera
    CameraInstance* getActiveCamera() const { return activeCamera.lock().get(); }

    // Context
    Context& getContext() const { return context; }
    Environment& getEnvironment() { return environment; }
    const Environment& getEnvironment() const { return environment; }
    RenderSettings& getRenderSettings() { return *renderSettings; }
    const RenderSettings& getRenderSettings() const { return *renderSettings; }

    // Dirty flags
    void setDirtyFlag(DirtyFlag flag) { dirtyFlags |= flag; }
    void clearDirtyFlag(DirtyFlag flag) { dirtyFlags &= ~flag; }
    bool isDirty(DirtyFlag flag) const { return (dirtyFlags & flag) != 0; }
    bool isAnyDirty() const { return dirtyFlags & (TLAS | Meshes | Textures); }
    void clearDirtyFlags() { dirtyFlags = 0; }
    void clearAccumulationDirtyFlag() { dirtyFlags &= ~Accumulation; }
};
