#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>
#include <string>
#include "CUDA/rstd/Vector.h"
#include "CUDA/rstd/UniquePtr.h"
#include "CUDA/Unique/SharedVector.h"
#include "Scene/RenderSettings.h"
#include "Mesh/MeshAsset.h"
#include "Mesh/GaussianAsset.h"
#include "Light/PointLight.h"
#include "Light/SpotLight.h"
#include "Light/RectLight.h"
#include "Light/DirectionalLight.h"
#include "Vulkan/Context.h"
#include "Scene/Texture.h"

#include <glm/mat4x4.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

using glm::ivec2;
using glm::mat4;
using glm::quat;
using glm::vec2;
using glm::vec3;
using glm::vec4;
using glm::angleAxis;
using glm::cross;
using glm::dot;
using glm::inverse;
using glm::length;
using glm::mat4_cast;
using glm::normalize;
using glm::perspective;
using glm::quat_cast;
using glm::radians;
using glm::transpose;

#include "Scene/Environment.h"

class SceneObject;
class MeshInstance;
class GaussianInstance;
class CameraInstance;
class LightInstance;

enum DirtyFlag : uint8_t {
    TLAS         = 1 << 0,
    Meshes       = 1 << 1,
    Textures     = 1 << 2,
    Accumulation = 1 << 3,
    EnvironmentCdf = 1 << 4,
    Lights       = 1 << 5,
    CameraState  = 1 << 6,
    GaussianData = 1 << 7,
};

class Scene {
    friend class LightInstance;
    Context& context;

    std::vector<Texture> textures;
    std::vector<std::string> textureNames;
    nr::rstd::vector<MeshAsset> meshAssets;
    nr::rstd::vector<GaussianAsset> gaussianAssets;
    nr::rstd::unique_ptr<Environment> environment;
    RenderSettings renderSettings{};

    // VMA owns these allocations. Host code uses the persistent VMA mapping,
    // while CUDA/OptiX uses the separately imported external-memory address.
    nr::cuda::SharedVector<PointLight> pointLights;
    nr::cuda::SharedVector<SpotLight> spotLights;
    nr::cuda::SharedVector<RectLight> rectLights;
    nr::cuda::SharedVector<DirectionalLight> directionalLights;

    // Render-ready Gaussian attributes shared with CUDA kernels.
    nr::rstd::vector<float> gaussianOpacities;
    nr::rstd::vector<glm::vec3> gaussianShCoeffs;

    std::vector<std::shared_ptr<SceneObject>> sceneObjects;
    std::vector<std::shared_ptr<GaussianInstance>> gaussianInstances;
    uint32_t gaussianCount{};

    std::shared_ptr<CameraInstance> viewportCamera;
    std::weak_ptr<CameraInstance> activeCamera;
    uint64_t activeCameraRevision{};
    uint64_t lightRevision{1};
    uint64_t activeObjectId = 0;
    uint64_t nextObjectId = 1;
    uint8_t dirtyFlags = 0;
    std::vector<uint32_t> dirtyMeshInstanceIndices;
    std::vector<uint32_t> dirtyGaussianInstanceIndices;
    std::vector<uint8_t> dirtyGaussianInstanceFlags;

    std::weak_ptr<SceneObject> copiedObject;
    std::function<void()> mutationBarrier;

    std::shared_ptr<SceneObject> findObjectPtr(const SceneObject* object) const;
    std::shared_ptr<SceneObject> findObjectPtr(uint64_t objectId) const;
    void activateCamera(const std::shared_ptr<CameraInstance>& camera);

    uint32_t registerObject(std::unique_ptr<SceneObject> sceneObject);
    void rebuildGaussianInstanceCache();
    bool remove(SceneObject* objToRemove);
    void reparent(SceneObject* objectToMove, SceneObject* newParent);
    std::shared_ptr<SceneObject> cloneHierarchy(const SceneObject* source);
    void notifyGeometryChanged();
public:
    Scene(Context& context);
    ~Scene();

    void setMutationBarrier(std::function<void()> barrier) {
        mutationBarrier = std::move(barrier);
    }
    void synchronizeBeforeMutation();

    void load(const std::string& path);
    void importFile(const std::string& path);
    void read(const std::string& path);

    // Object lifetime
    void clear();
    uint64_t add(std::unique_ptr<SceneObject> sceneObject);
    uint32_t add(MeshAsset meshAsset);
    uint32_t add(GaussianAsset gaussianAsset);
    Texture& add(Texture&& texture);
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
    uint32_t getActiveMeshInstanceIndex() const;
    MeshAsset* getMeshAsset(const std::string& name);
    const MeshAsset* getMeshAsset(const std::string& name) const;
    MeshAsset& getMeshAsset(uint32_t index) { return meshAssets[index]; }
    const MeshAsset& getMeshAsset(uint32_t index) const { return meshAssets[index]; }
    const nr::rstd::vector<MeshAsset>& getMeshAssets() const { return meshAssets; }
    nr::rstd::vector<MeshAsset>& getMeshAssets() { return meshAssets; }

    // Gaussian assets
    GaussianAsset& getGaussianAsset(uint32_t index) { return gaussianAssets[index]; }
    const GaussianAsset& getGaussianAsset(uint32_t index) const { return gaussianAssets[index]; }
    const nr::rstd::vector<GaussianAsset>& getGaussianAssets() const { return gaussianAssets; }
    nr::rstd::vector<GaussianAsset>& getGaussianAssets() { return gaussianAssets; }
    const std::vector<std::shared_ptr<GaussianInstance>>& getGaussianInstances() const {
        return gaussianInstances;
    }
    uint32_t getGaussianCount() const { return gaussianCount; }
    void buildGaussianRenderData();
    const float* getGaussianOpacities() const { return gaussianOpacities.data(); }
    const glm::vec3* getGaussianShCoeffs() const { return gaussianShCoeffs.data(); }
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
    std::shared_ptr<CameraInstance> getActiveCameraPtr() const { return activeCamera.lock(); }
    uint64_t getActiveCameraRevision() const { return activeCameraRevision; }
    CameraInstance* getRenderCamera() const {
        if (auto camera = activeCamera.lock())
            return camera.get();
        return viewportCamera.get();
    }
    bool setActiveCamera(CameraInstance* camera);

    // Light GPU data, Vulkan/CUDA-shared. Host pointers (below) are for CPU-side use
    // (UI, transform updates, host-side selection-weight sums); the *Device variants
    // are the CUDA-mapped view and are the only ones safe to dereference in kernels.
    const PointLight* getPointLights() const { return pointLights.data(); }
    const SpotLight* getSpotLights() const { return spotLights.data(); }
    const RectLight* getRectLights() const { return rectLights.data(); }
    const DirectionalLight* getDirectionalLights() const { return directionalLights.data(); }
    const PointLight* getPointLightsDevice() const { return pointLights.devicePointer(); }
    const SpotLight* getSpotLightsDevice() const { return spotLights.devicePointer(); }
    const RectLight* getRectLightsDevice() const { return rectLights.devicePointer(); }
    const DirectionalLight* getDirectionalLightsDevice() const { return directionalLights.devicePointer(); }
    uint32_t getPointLightCount() const { return static_cast<uint32_t>(pointLights.size()); }
    uint32_t getSpotLightCount() const { return static_cast<uint32_t>(spotLights.size()); }
    uint32_t getRectLightCount() const { return static_cast<uint32_t>(rectLights.size()); }
    uint32_t getDirectionalLightCount() const { return static_cast<uint32_t>(directionalLights.size()); }
    uint64_t getLightRevision() const { return lightRevision; }

    // Light registration (called by Scene internals)
    uint32_t registerLight(LightInstance& light);
    void unregisterLight(LightInstance& light);

    // Context
    Context& getContext() const { return context; }
    Environment& getEnvironment() { return *environment; }
    const Environment& getEnvironment() const { return *environment; }
    RenderSettings& getRenderSettings() { return renderSettings; }
    const RenderSettings& getRenderSettings() const { return renderSettings; }

    // Dirty flags
    void setDirtyFlag(DirtyFlag flag) {
        dirtyFlags |= flag;
        if (flag == Lights)
            ++lightRevision;
    }
    void clearDirtyFlag(DirtyFlag flag) { dirtyFlags &= ~flag; }
    bool isDirty(DirtyFlag flag) const { return (dirtyFlags & flag) != 0; }
    bool isAnyDirty() const { return dirtyFlags & (TLAS | Meshes | Textures | EnvironmentCdf | Lights | CameraState); }
    void clearDirtyFlags() { dirtyFlags = 0; }
    void clearAccumulationDirtyFlag() { dirtyFlags &= ~Accumulation; }

    // Per-mesh-instance transform dirty tracking, so a single object move can
    // be applied to the TLAS without re-baking every instance in the scene.
    // Only set for pure transform edits (see MeshInstance::onTransformUpdated);
    // structural changes (add/remove) go through the TLAS dirty flag alone and
    // fall back to a full rebuild.
    void markMeshInstanceTransformDirty(uint32_t instanceIndex);
    const std::vector<uint32_t>& getDirtyMeshInstanceIndices() const { return dirtyMeshInstanceIndices; }
    void clearDirtyMeshInstanceIndices() { dirtyMeshInstanceIndices.clear(); }
    uint32_t getMeshInstanceIndex(const SceneObject* object) const;
    void markGaussianInstanceTransformDirty(uint32_t instanceIndex);
    const std::vector<uint32_t>& getDirtyGaussianInstanceIndices() const {
        return dirtyGaussianInstanceIndices;
    }
    void clearDirtyGaussianInstanceIndices() {
        dirtyGaussianInstanceIndices.clear();
        std::fill(dirtyGaussianInstanceFlags.begin(), dirtyGaussianInstanceFlags.end(), uint8_t{0});
    }
};
