#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>
#include <string>
#include "CUDA/rstd/Vector.h"
#include "CUDA/rstd/UniquePtr.h"
#include "CUDA/Unique/SharedVector.h"
#include "Scene/Handle.h"
#include "Scene/RenderSettings.h"
#include "Scene/SceneResources.h"
#include "Mesh/Assets/MeshAsset.h"
#include "Mesh/Assets/GaussianAsset.h"
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

    // Every resource is reference counted: Scene::add hands out an owning
    // reference, instances and assets keep the references they use alive, and
    // the moment the last one is dropped the resource releases its device
    // memory and its slot is recycled. Slot indices stay stable while a
    // resource lives, which is what lets GPU-side data address them by index.
    //
    // Declaration order is destruction order reversed, and references must not
    // outlive the registry they point into: textures are referenced by
    // materials, materials by mesh assets, and all of them by scene objects
    // (declared further down), so each holder is declared after its target.
    TextureRegistry textures;
    MaterialRegistry materials;
    // Textures sampled by each material slot. Materials are plain GPU structs
    // holding bare texture indices, so the Scene carries their ownership.
    std::vector<std::vector<TextureRef>> materialTextures;
    TextureRef environmentTexture;
    MeshAssetRegistry meshAssets;
    GaussianAssetRegistry gaussianAssets;
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
    // Coefficient-major RGB binary16 values. Opacity remains float because it
    // directly controls stochastic acceptance and benefits less from packing.
    nr::rstd::vector<__half> gaussianShCoeffs;
    nr::rstd::vector<uint32_t> gaussianInstanceOffsets;
    uint32_t gaussianShCoefficientCount{MaxSphericalHarmonicsCoefficientCount};

    // Objects live in a dense array so iteration and the TLAS build stay
    // cache friendly; the sparse slot table beside it gives every object a
    // stable, generation-checked handle and O(1) lookup.
    struct ObjectSlot
    {
        uint32_t denseIndex{~0u};
        uint32_t generation{};
    };
    std::vector<std::shared_ptr<SceneObject>> sceneObjects;
    std::vector<ObjectSlot> objectSlots;
    std::vector<uint32_t> freeObjectSlots;
    std::vector<std::shared_ptr<GaussianInstance>> gaussianInstances;
    uint32_t gaussianCount{};

    std::shared_ptr<CameraInstance> viewportCamera;
    std::weak_ptr<CameraInstance> activeCamera;
    uint64_t activeCameraRevision{};
    uint64_t lightRevision{1};
    SceneObjectHandle activeObject;
    uint8_t dirtyFlags = 0;
    std::vector<uint32_t> dirtyMeshInstanceIndices;
    std::vector<uint32_t> dirtyGaussianInstanceIndices;
    std::vector<uint8_t> dirtyGaussianInstanceFlags;

    std::weak_ptr<SceneObject> copiedObject;
    std::function<void()> mutationBarrier;
    std::atomic<bool> gpuSyncPending_{false};

    std::shared_ptr<SceneObject> findObjectPtr(const SceneObject* object) const;
    std::shared_ptr<SceneObject> findObjectPtr(SceneObjectHandle handle) const;
    void activateCamera(const std::shared_ptr<CameraInstance>& camera);

    SceneObjectHandle allocateObjectSlot(uint32_t denseIndex);
    void releaseObjectSlot(SceneObjectHandle handle);
    uint32_t registerObject(std::unique_ptr<SceneObject> sceneObject);
    void rebuildGaussianInstanceCache();
    void retainMaterialTextures(MaterialHandle handle, const Material& material);
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
    // Returns true if a GPU sync is needed before the next render frame.
    // The caller (render thread) should sync its stream when this is true.
    bool consumeGpuSync() { return gpuSyncPending_.exchange(false); }
    // Drops the Scene-held references of resources that other resources
    // reclaimed on their own. Runs before every mutation; exposed so callers
    // that want the memory back at a specific point can ask for it.
    void reclaimUnusedResources();

    void load(const std::string& path);
    void importFile(const std::string& path);
    void read(const std::string& path);

    // Object lifetime
    void clear();
    SceneObjectHandle add(std::unique_ptr<SceneObject> sceneObject);
    bool removeObject(SceneObjectHandle handle);
    bool replaceObject(SceneObject* oldObject, std::unique_ptr<SceneObject> newObject);

    // Resource lifetime. The returned reference owns the resource: it stays
    // alive for as long as any reference to it does, and releases its GPU
    // memory as soon as the last one is dropped.
    MeshAssetRef add(MeshAsset meshAsset);
    MaterialRef add(Material material);
    GaussianAssetRef add(GaussianAsset gaussianAsset);
    TextureRef add(Texture texture);
    void updateMaterial(MaterialHandle handle, const Material& material);

    // Hierarchy
    bool reparentObject(SceneObjectHandle handle, SceneObjectHandle newParent = {});

    // Clipboard
    void copyObject(SceneObjectHandle handle);
    void paste();

    // Lookup
    bool isValid(SceneObjectHandle handle) const;
    SceneObject* getObject(SceneObjectHandle handle) const { return findObjectPtr(handle).get(); }
    std::shared_ptr<SceneObject> getObjectPtr(SceneObjectHandle handle) const { return findObjectPtr(handle); }
    const std::vector<std::shared_ptr<SceneObject>>& getSceneObjects() const { return sceneObjects; }
    std::vector<std::shared_ptr<SceneObject>> getRootObjects() const;
    std::vector<std::shared_ptr<MeshInstance>> getMeshInstances() const;
    uint32_t getActiveCryptomatteId(uint32_t selectedGaussianIndex) const;
    MeshAssetHandle findMeshAsset(const std::string& path) const;
    MeshAsset* getMeshAsset(MeshAssetHandle handle) { return meshAssets.find(handle); }
    const MeshAsset* getMeshAsset(MeshAssetHandle handle) const { return meshAssets.find(handle); }
    MeshAssetRef getMeshAssetRef(MeshAssetHandle handle) { return {meshAssets, handle}; }
    const nr::rstd::vector<MeshAsset>& getMeshAssets() const { return meshAssets.storage(); }
    nr::rstd::vector<MeshAsset>& getMeshAssets() { return meshAssets.storage(); }
    const nr::rstd::vector<Material>& getMaterials() const { return materials.storage(); }
    nr::rstd::vector<Material>& getMaterials() { return materials.storage(); }
    const Material& getMaterial(MaterialHandle handle) const { return materials[handle]; }
    Material& getMaterial(MaterialHandle handle) { return materials[handle]; }
    // Gaussian assets
    GaussianAsset* getGaussianAsset(GaussianAssetHandle handle) { return gaussianAssets.find(handle); }
    const GaussianAsset* getGaussianAsset(GaussianAssetHandle handle) const { return gaussianAssets.find(handle); }
    GaussianAssetRef getGaussianAssetRef(GaussianAssetHandle handle) { return {gaussianAssets, handle}; }
    const nr::rstd::vector<GaussianAsset>& getGaussianAssets() const { return gaussianAssets.storage(); }
    nr::rstd::vector<GaussianAsset>& getGaussianAssets() { return gaussianAssets.storage(); }
    const GaussianAssetRegistry& getGaussianAssetRegistry() const { return gaussianAssets; }
    const std::vector<std::shared_ptr<GaussianInstance>>& getGaussianInstances() const {
        return gaussianInstances;
    }
    uint32_t getGaussianCount() const { return gaussianCount; }
    void buildGaussianRenderData();
    const float* getGaussianOpacities() const { return gaussianOpacities.data(); }
    const __half* getGaussianShCoeffs() const { return gaussianShCoeffs.data(); }
    const uint32_t* getGaussianInstanceOffsets() const { return gaussianInstanceOffsets.data(); }
    uint32_t getGaussianShCoefficientCount() const { return gaussianShCoefficientCount; }
    const std::vector<Texture>& getTextures() const { return textures.storage(); }
    const TextureRegistry& getTextureRegistry() const { return textures; }
    const Texture* getTexture(TextureHandle handle) const { return textures.find(handle); }
    // One entry per texture slot; released slots read as empty names.
    std::vector<std::string> getTextureNames() const;
    void setEnvironmentTexture(const TextureRef& texture);
    void clearEnvironmentTexture();

    // Active object
    void setActiveObject(SceneObjectHandle handle) { activeObject = handle; }
    void clearActiveObject() { activeObject = {}; }
    SceneObjectHandle getActiveObjectHandle() const { return activeObject; }
    SceneObject* getActiveObject() const { return findObjectPtr(activeObject).get(); }
    std::shared_ptr<SceneObject> getActiveObjectPtr() const { return findObjectPtr(activeObject); }

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
    bool isAnyDirty() const { return dirtyFlags & (TLAS | Meshes | Textures
        | EnvironmentCdf | Lights | CameraState | GaussianData); }
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
