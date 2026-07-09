#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include <string>
#include "CUDA/Texture.h"
#include "CUDA/rstd/Vector.h"
#include "Scene/GpuInstance.h"
#include "Scene/RenderSettings.h"
#include "Mesh/MeshAsset.h"
#include "Mesh/GaussianAsset.h"
#include "Light/PointLight.h"
#include "Light/SpotLight.h"
#include "Light/RectLight.h"
#include "Light/DirectionalLight.h"
#include "Vulkan/Context.h"
#include "Vulkan/Texture.h"
#include <vulkan/vulkan.hpp>

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
class Buffer;

enum DirtyFlag : uint8_t {
    TLAS         = 1 << 0,
    Meshes       = 1 << 1,
    Textures     = 1 << 2,
    Accumulation = 1 << 3,
    EnvironmentCdf = 1 << 4,
    Lights       = 1 << 5,
};

class Scene {
    friend class LightInstance;
    Context& context;

    std::vector<Texture> textures;
    std::vector<std::string> textureNames;
    nr::rstd::vector<MeshAsset> meshAssets;
    nr::rstd::vector<GaussianAsset> gaussianAssets;
    Environment* environment{};
    RenderSettings renderSettings{};

    // GPU data — unified memory arrays
    nr::rstd::vector<PointLight> pointLights;
    nr::rstd::vector<SpotLight> spotLights;
    nr::rstd::vector<RectLight> rectLights;
    nr::rstd::vector<DirectionalLight> directionalLights;
    nr::rstd::vector<GpuInstance> gpuInstances;
    nr::rstd::vector<float> gaussianOpacities;
    nr::rstd::vector<glm::vec3> gaussianSpectrumCoeffs;
    nr::rstd::vector<CudaTexture> cudaTextures;

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
    void clear();
    uint64_t add(std::unique_ptr<SceneObject> sceneObject);
    uint32_t add(MeshAsset meshAsset);
    uint32_t add(GaussianAsset gaussianAsset);
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
    std::vector<std::shared_ptr<GaussianInstance>> getGaussianInstances() const;
    uint32_t getGaussianCount() const;
    void buildGaussianRenderData();
    const float* getGaussianOpacities() const { return gaussianOpacities.data(); }
    const glm::vec3* getGaussianSpectrumCoeffs() const { return gaussianSpectrumCoeffs.data(); }
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

    // Light GPU data (unified memory, one array per type)
    const PointLight* getPointLights() const { return pointLights.data(); }
    const SpotLight* getSpotLights() const { return spotLights.data(); }
    const RectLight* getRectLights() const { return rectLights.data(); }
    const DirectionalLight* getDirectionalLights() const { return directionalLights.data(); }
    uint32_t getPointLightCount() const { return static_cast<uint32_t>(pointLights.size()); }
    uint32_t getSpotLightCount() const { return static_cast<uint32_t>(spotLights.size()); }
    uint32_t getRectLightCount() const { return static_cast<uint32_t>(rectLights.size()); }
    uint32_t getDirectionalLightCount() const { return static_cast<uint32_t>(directionalLights.size()); }

    // Gpu instances
    const GpuInstance* getGpuInstances() const { return gpuInstances.data(); }
    uint32_t getGpuInstanceCount() const { return static_cast<uint32_t>(gpuInstances.size()); }
    nr::rstd::vector<GpuInstance>& getGpuInstancesRef() { return gpuInstances; }

    // Cuda textures
    const CudaTexture* getCudaTextures() const { return cudaTextures.data(); }
    uint32_t getCudaTextureCount() const { return static_cast<uint32_t>(cudaTextures.size()); }
    nr::rstd::vector<CudaTexture>& getCudaTexturesRef() { return cudaTextures; }

    // Light registration (called by Scene internals)
    uint32_t registerLight(LightInstance& light);
    void unregisterLight(const LightInstance& light);

    // Context
    Context& getContext() const { return context; }
    Environment& getEnvironment() { return *environment; }
    const Environment& getEnvironment() const { return *environment; }
    RenderSettings& getRenderSettings() { return renderSettings; }
    const RenderSettings& getRenderSettings() const { return renderSettings; }

    // Dirty flags
    void setDirtyFlag(DirtyFlag flag) { dirtyFlags |= flag; }
    void clearDirtyFlag(DirtyFlag flag) { dirtyFlags &= ~flag; }
    bool isDirty(DirtyFlag flag) const { return (dirtyFlags & flag) != 0; }
    bool isAnyDirty() const { return dirtyFlags & (TLAS | Meshes | Textures | EnvironmentCdf | Lights); }
    void clearDirtyFlags() { dirtyFlags = 0; }
    void clearAccumulationDirtyFlag() { dirtyFlags &= ~Accumulation; }
};
