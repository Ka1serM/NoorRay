#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>
#include <string>
#include <vector>
#include <memory>
#include "Scene/Handle.h"
#include "Scene/RenderSettings.h"
#include "Scene/Resources/SceneResources.h"
#include "Geometry/Mesh/Assets/MeshAsset.h"
#include "Geometry/Mesh/Assets/GaussianAsset.h"
#include "Rendering/Lighting/PointLight.h"
#include "Rendering/Lighting/SpotLight.h"
#include "Rendering/Lighting/RectLight.h"
#include "Rendering/Lighting/DirectionalLight.h"
#include "Scene/Resources/Texture.h"

#include <glm/mat4x4.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

namespace MaterialX_v1_39_4
{
class Document;
using DocumentPtr = std::shared_ptr<Document>;
}
namespace MaterialX = MaterialX_v1_39_4;

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

#include "Scene/Resources/Environment.h"

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

    // Every resource is reference counted: Scene::add hands out an owning
    // reference, instances and assets keep the references they use alive, and
    // the moment the last one is dropped the resource releases its device
    // memory and its slot is recycled. Slot indices stay stable while a
    // resource lives, which is what lets GPU-side data address them by index.
    //
    // Textures are scene-owned assets rather than reference-counted resources.
    // Handles carry the scene generation so references from an old file can
    // never alias a texture in the next file.
    std::vector<Texture> textures;
    uint32_t textureGeneration_{1};
    uint64_t textureRevision_{1};
    MaterialRegistry materials;
    // Scene ownership keeps materials available in the global material list
    // even when no mesh slot currently references them.
    std::vector<MaterialRef> materialOwners;
    MeshAssetRegistry meshAssets;
    GaussianAssetRegistry gaussianAssets;

    // Per-material MaterialX source file paths. Parallel to the materials
    // vector: entry i is the .mtlx path for materials[i]. Empty means the
    // material's document is held in memory (materialxDocuments[i]) instead.
    std::vector<std::string> materialxSourcePaths;
    // In-memory MaterialX documents, one per material slot (parallel to the
    // materials vector). MaterialX XML is only ever parsed to a Document here
    // (import) or serialized back to XML (export); the document itself is
    // what lives in the app. Null entries mean the material is either backed
    // by materialxSourcePaths[i] or is an un-authored slot whose document is
    // lowered on demand from the default material.
    std::vector<MaterialX::DocumentPtr> materialxDocuments;
    uint32_t selectedMaterialSlot{0};

    std::unique_ptr<Environment> environment;
    RenderSettings renderSettings{};

    // Lighting data is host-owned and uploaded by the native Vulkan renderer.
    std::vector<PointLight> pointLights;
    std::vector<SpotLight> spotLights;
    std::vector<RectLight> rectLights;
    std::vector<DirectionalLight> directionalLights;

    // Render-ready Gaussian attributes shared with the Vulkan renderer.
    std::vector<float> gaussianOpacities;
    // Coefficient-major RGB binary16 values. Opacity remains float because it
    // directly controls stochastic acceptance and benefits less from packing.
    std::vector<uint16_t> gaussianShCoeffs;
    std::vector<uint32_t> gaussianInstanceOffsets;
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

    // Scene-wide identity indexes. Meshes use their asset path/name and
    // textures use their source path/name. The values are handles only; the
    // actual resources remain owned by the scene containers above.
    std::unordered_map<std::string, MeshAssetHandle> meshAssetsByPath_;
    std::unordered_map<std::string, TextureHandle> texturesByKey_;

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

    // Resolved file path -> root object of the hierarchy built the first time
    // SceneImporter imported that path. A repeat import of the same path
    // clones this hierarchy (see cloneHierarchy) instead of re-parsing the
    // file and re-uploading its meshes/textures. Keyed by the fully resolved
    // path so relative-vs-absolute spellings of the same file still collide.
    std::unordered_map<std::string, SceneObjectHandle> importedFileRoots_;

    std::shared_ptr<SceneObject> findObjectPtr(const SceneObject* object) const;
    std::shared_ptr<SceneObject> findObjectPtr(SceneObjectHandle handle) const;
    void activateCamera(const std::shared_ptr<CameraInstance>& camera);

    SceneObjectHandle allocateObjectSlot(uint32_t denseIndex);
    void releaseObjectSlot(SceneObjectHandle handle);
    uint32_t registerObject(std::unique_ptr<SceneObject> sceneObject);
    void rebuildGaussianInstanceCache();
    bool remove(SceneObject* objToRemove);
    void reparent(SceneObject* objectToMove, SceneObject* newParent);
    void notifyGeometryChanged();
public:
    Scene();
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

    // Resource lifetime. Meshes/materials use reference-counted lifetime, but
    // textures are part of the scene-wide image library: Scene owns every
    // texture added here until clear() opens a new file. Returned handles can
    // therefore be used freely by materials, MaterialX nodes, and the
    // environment without an importer-local owner.
    // Reuses an existing asset with the same path/name by default. Import
    // paths that intentionally alter the material can opt out.
    MeshAssetRef add(MeshAsset meshAsset, bool reuseExisting = true);
    MaterialRef add(Material material);
    // Adds a native material. Importers that only carry the simple authoring
    // record lower it to a canonical MaterialX document first
    // (nr::materialx::documentFromAuthoring) and pass the resulting document
    // here, so every material compiles through the same MaterialX -> SVM
    // pipeline as authored graphs. A null document is allowed: it means an
    // un-authored slot whose document is lowered on demand from the default
    // material.
    MaterialRef addMaterial(MaterialX::DocumentPtr material);
    // Replaces the document of an existing material slot in place, leaving the
    // slot's compiled program untouched. Used by live-editing paths (Hydra)
    // that republish a document while a replacement compiles in the background.
    // A null document clears the slot's authored graph (the default MaterialX
    // material is lowered on demand).
    void updateMaterialDocument(MaterialHandle handle, MaterialX::DocumentPtr document);
    GaussianAssetRef add(GaussianAsset gaussianAsset);
    // Adds an image to the scene-wide texture library. Scene owns it until
    // clear(); callers retain only this stable, non-owning registry handle.
    TextureHandle addTexture(Texture texture);
    void updateMaterial(MaterialHandle handle, const Material& material);
    void invalidateMaterial(MaterialHandle handle);
    // Pre-allocates the registries and dense object arrays used by an import
    // batch. Call this after parsing/preparing payloads and before publishing
    // them serially to avoid managed-storage relocation for every asset.
    void reserveForImport(
        size_t meshCount, size_t materialCount, size_t objectCount = 0);

    // Hierarchy
    bool reparentObject(SceneObjectHandle handle, SceneObjectHandle newParent = {});

    // Clipboard
    void copyObject(SceneObjectHandle handle);
    void paste();
    // Deep-copies source and its children, sharing (not duplicating) every
    // mesh/material/texture/Gaussian resource the originals reference -- the
    // clones are new SceneObjects/MeshInstances, not new GPU uploads. Used by
    // paste() and by SceneImporter's file-level import cache to instance a
    // previously imported file without re-parsing or re-uploading it.
    std::shared_ptr<SceneObject> cloneHierarchy(const SceneObject* source);

    // Lookup
    bool isValid(SceneObjectHandle handle) const;
    SceneObject* getObject(SceneObjectHandle handle) const { return findObjectPtr(handle).get(); }
    std::shared_ptr<SceneObject> getObjectPtr(SceneObjectHandle handle) const { return findObjectPtr(handle); }
    const std::vector<std::shared_ptr<SceneObject>>& getSceneObjects() const { return sceneObjects; }
    std::vector<std::shared_ptr<SceneObject>> getRootObjects() const;
    std::vector<std::shared_ptr<MeshInstance>> getMeshInstances() const;
    uint32_t getActiveCryptomatteId(uint32_t selectedGaussianIndex) const;
    TextureHandle findTexture(const std::string& key) const;
    MeshAssetHandle findMeshAsset(const std::string& path) const;
    // Returns the root of a previously imported file's hierarchy (see
    // importedFileRoots_), or an invalid handle if this path has never been
    // imported or that hierarchy was since removed.
    SceneObjectHandle findImportedFileRoot(const std::string& resolvedPath) const;
    void registerImportedFileRoot(
        const std::string& resolvedPath, SceneObjectHandle handle);
    MeshAsset* getMeshAsset(MeshAssetHandle handle) { return meshAssets.find(handle); }
    const MeshAsset* getMeshAsset(MeshAssetHandle handle) const { return meshAssets.find(handle); }
    MeshAssetRef getMeshAssetRef(MeshAssetHandle handle) { return {meshAssets, handle}; }
    const std::vector<MeshAsset>& getMeshAssets() const { return meshAssets.storage(); }
    std::vector<MeshAsset>& getMeshAssets() { return meshAssets.storage(); }
    const std::vector<Material>& getMaterials() const { return materials.storage(); }
    std::vector<Material>& getMaterials() { return materials.storage(); }
    const MaterialRegistry& getMaterialRegistry() const { return materials; }
    MaterialRef getMaterialRef(const MaterialHandle handle) { return {materials, handle}; }
    const Material& getMaterial(MaterialHandle handle) const { return materials[handle]; }
    Material& getMaterial(MaterialHandle handle) { return materials[handle]; }
    // Gaussian assets
    GaussianAsset* getGaussianAsset(GaussianAssetHandle handle) { return gaussianAssets.find(handle); }
    const GaussianAsset* getGaussianAsset(GaussianAssetHandle handle) const { return gaussianAssets.find(handle); }
    GaussianAssetRef getGaussianAssetRef(GaussianAssetHandle handle) { return {gaussianAssets, handle}; }
    const std::vector<GaussianAsset>& getGaussianAssets() const { return gaussianAssets.storage(); }
    std::vector<GaussianAsset>& getGaussianAssets() { return gaussianAssets.storage(); }
    const GaussianAssetRegistry& getGaussianAssetRegistry() const { return gaussianAssets; }
    const std::vector<std::shared_ptr<GaussianInstance>>& getGaussianInstances() const {
        return gaussianInstances;
    }
    uint32_t getGaussianCount() const { return gaussianCount; }
    void buildGaussianRenderData();
    const float* getGaussianOpacities() const { return gaussianOpacities.data(); }
    const uint16_t* getGaussianShCoeffs() const { return gaussianShCoeffs.data(); }
    const uint32_t* getGaussianInstanceOffsets() const { return gaussianInstanceOffsets.data(); }
    uint32_t getGaussianShCoefficientCount() const { return gaussianShCoefficientCount; }
    const std::vector<Texture>& getTextures() const { return textures; }
    TextureHandle getTextureHandle(uint32_t index) const {
        return index < textures.size()
            ? TextureHandle(index, textureGeneration_) : TextureHandle{};
    }
    uint64_t getTextureRevision() const { return textureRevision_; }
    const Texture* getTexture(TextureHandle handle) const {
        return handle.isValid() && handle.generation() == textureGeneration_
            && handle.index() < textures.size() ? &textures[handle.index()] : nullptr;
    }
    // One entry per scene-owned texture.
    std::vector<std::string> getTextureNames() const;
    void setEnvironmentTexture(TextureHandle texture);
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

    // Light GPU data, uploaded by Vulkan. Host pointers (below) are for CPU-side use
    // (UI, transform updates, host-side selection-weight sums); the *Device variants
    // are host views used to stage native GPU buffers.
    const PointLight* getPointLights() const { return pointLights.data(); }
    const SpotLight* getSpotLights() const { return spotLights.data(); }
    const RectLight* getRectLights() const { return rectLights.data(); }
    const DirectionalLight* getDirectionalLights() const { return directionalLights.data(); }
    const PointLight* getPointLightsDevice() const { return pointLights.data(); }
    const SpotLight* getSpotLightsDevice() const { return spotLights.data(); }
    const RectLight* getRectLightsDevice() const { return rectLights.data(); }
    const DirectionalLight* getDirectionalLightsDevice() const { return directionalLights.data(); }
    uint32_t getPointLightCount() const { return static_cast<uint32_t>(pointLights.size()); }
    uint32_t getSpotLightCount() const { return static_cast<uint32_t>(spotLights.size()); }
    uint32_t getRectLightCount() const { return static_cast<uint32_t>(rectLights.size()); }
    uint32_t getDirectionalLightCount() const { return static_cast<uint32_t>(directionalLights.size()); }
    uint64_t getLightRevision() const { return lightRevision; }

    // Light registration (called by Scene internals)
    uint32_t registerLight(LightInstance& light);
    void unregisterLight(LightInstance& light);

    const std::vector<std::string>& getMaterialXSourcePaths() const { return materialxSourcePaths; }
    std::vector<std::string>& getMaterialXSourcePaths() { return materialxSourcePaths; }
    const std::vector<MaterialX::DocumentPtr>& getMaterialXDocuments() const { return materialxDocuments; }
    std::vector<MaterialX::DocumentPtr>& getMaterialXDocuments() { return materialxDocuments; }
    uint32_t getSelectedMaterialSlot() const { return selectedMaterialSlot; }
    void setSelectedMaterialSlot(const uint32_t slot) { selectedMaterialSlot = slot; }

    // Context
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
