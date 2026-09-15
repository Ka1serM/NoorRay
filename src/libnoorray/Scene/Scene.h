#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <unordered_map>
#include <string>
#include <vector>
#include <MaterialXCore/Document.h>
#include "Scene/Handle.h"
#include "Shared/RenderSettings.h"
#include "Mesh/Assets/Mesh.h"
#include "Mesh/Assets/Gaussian.h"
#include "Shared/Light.h"
#include "Texture/Texture.h"
#include "Shared/Math.h"

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

#include "Environment/Environment.h"

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

    std::deque<Texture> textures;
    uint64_t textureRevision_{1};
    uint64_t materialRevision_{1};
    std::deque<Material> materials;
    std::deque<Mesh> meshes;
    std::deque<GaussianAsset> gaussianAssets;

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

    // Lighting data is host-owned and uploaded by the native graphics API renderer.
    std::vector<PointLight> pointLights;
    std::vector<SpotLight> spotLights;
    std::vector<RectLight> rectLights;
    std::vector<DirectionalLight> directionalLights;

    // Render-ready Gaussian attributes shared with the graphics API renderer.
    std::vector<float> gaussianOpacities;
    // Coefficient-major RGB binary16 values. Opacity remains float because it
    // directly controls stochastic acceptance and benefits less from packing.
    std::vector<half> gaussianShCoeffs;
    std::vector<uint32_t> gaussianInstanceOffsets;
    uint32_t gaussianShCoefficientCount{MaxSphericalHarmonicsCoefficientCount};

    // Objects live in a dense array so iteration and the TLAS build stay cache
    // friendly. Each slot records where its object currently sits, which is
    // what gives SceneObjectHandle a stable, generation-checked identity.
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

    std::unordered_map<std::string, Mesh*> meshesByPath_;
    std::unordered_map<std::string, Texture*> texturesByKey_;

    std::shared_ptr<CameraInstance> viewportCamera;
    std::weak_ptr<CameraInstance> activeCamera;
    uint64_t activeCameraRevision{};
    uint64_t lightRevision{1};
    SceneObjectHandle activeObject;
    uint8_t dirtyFlags = 0;

    std::weak_ptr<SceneObject> copiedObject;
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
    void notifyMaterialChanged() { ++materialRevision_; }
public:
    Scene();
    ~Scene();

    // Marks that CPU-side scene edits must be published before the next GPU
    // snapshot. The session coalesces many UI edits into one wait at the
    // publication boundary.
    void synchronizeBeforeMutation();
    // Releases backend allocations without changing scene contents.
    void releaseGpuResources();
    // Returns true if a GPU sync is needed before the next render frame.
    // The caller (render thread) should sync its stream when this is true.
    bool consumeGpuSync() { return gpuSyncPending_.exchange(false); }
    void load(const std::string& path);
    void importFile(const std::string& path);
    void read(const std::string& path);

    // Object lifetime
    void clear();
    SceneObjectHandle add(std::unique_ptr<SceneObject> sceneObject);
    bool removeObject(SceneObjectHandle handle);
    bool replaceObject(SceneObject* oldObject, std::unique_ptr<SceneObject> newObject);

    Mesh* add(Mesh mesh, bool reuseExisting = true);
    Material* add(Material material);
    // Adds a native material. Importers that only carry the simple SVM
    // material record lower it to a canonical MaterialX document first
    // (nr::materialx::documentFromSvmMaterial) and pass the resulting document
    // here, so every material compiles through the same MaterialX -> SVM
    // pipeline as authored graphs. A null document is allowed: it means an
    // un-authored slot whose document is lowered on demand from the default
    // material.
    Material* addMaterial(MaterialX::DocumentPtr material);
    // Replaces the document of an existing material slot in place, leaving the
    // slot's compiled program untouched. Used by live-editing paths (Hydra)
    // that republish a document while a replacement compiles in the background.
    // A null document clears the slot's authored graph (the default MaterialX
    // material is lowered on demand).
    void updateMaterialDocument(Material* material, MaterialX::DocumentPtr document);
    GaussianAsset* add(GaussianAsset gaussianAsset);
    Texture* addTexture(Texture texture);
    void invalidateMaterial(Material* material);
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
    Texture* findTexture(const std::string& key) const;
    Mesh* findMesh(const std::string& path) const;
    // Returns the root of a previously imported file's hierarchy (see
    // importedFileRoots_), or an invalid handle if this path has never been
    // imported or that hierarchy was since removed.
    SceneObjectHandle findImportedFileRoot(const std::string& resolvedPath) const;
    void registerImportedFileRoot(
        const std::string& resolvedPath, SceneObjectHandle handle);
    const std::deque<Mesh>& getMeshes() const { return meshes; }
    std::deque<Mesh>& getMeshes() { return meshes; }
    const std::deque<Material>& getMaterials() const { return materials; }
    std::deque<Material>& getMaterials() { return materials; }
    // Publishes a freshly compiled program for one material and uploads that
    // material's own GPU allocations. No other material is touched.
    void setMaterialProgram(std::size_t materialIndex,
        nr::svm::CompiledSvmProgram program);
    uint32_t getMaterialIndex(const Material* material) const;
    const Material& getMaterial(const Material* material) const { return *material; }
    Material& getMaterial(Material* material) { return *material; }
    const std::deque<GaussianAsset>& getGaussianAssets() const { return gaussianAssets; }
    std::deque<GaussianAsset>& getGaussianAssets() { return gaussianAssets; }
    const std::vector<std::shared_ptr<GaussianInstance>>& getGaussianInstances() const {
        return gaussianInstances;
    }
    uint32_t getGaussianCount() const { return gaussianCount; }
    void buildGaussianRenderData();
    const float* getGaussianOpacities() const { return gaussianOpacities.data(); }
    const half* getGaussianShCoeffs() const { return gaussianShCoeffs.data(); }
    const uint32_t* getGaussianInstanceOffsets() const { return gaussianInstanceOffsets.data(); }
    uint32_t getGaussianShCoefficientCount() const { return gaussianShCoefficientCount; }
    const std::deque<Texture>& getTextures() const { return textures; }
    std::deque<Texture>& getTextures() { return textures; }
    Texture* getTexture(uint32_t index) {
        return index < textures.size() ? &textures[index] : nullptr;
    }
    const Texture* getTexture(uint32_t index) const {
        return index < textures.size() ? &textures[index] : nullptr;
    }
    uint64_t getTextureRevision() const { return textureRevision_; }
    uint64_t getMaterialRevision() const { return materialRevision_; }
    // One entry per scene-owned texture.
    std::vector<std::string> getTextureNames() const;
    void setEnvironmentTexture(Texture* texture);
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

    // Host-side light data, uploaded by the native renderer.
    const PointLight* getPointLights() const { return pointLights.data(); }
    const SpotLight* getSpotLights() const { return spotLights.data(); }
    const RectLight* getRectLights() const { return rectLights.data(); }
    const DirectionalLight* getDirectionalLights() const { return directionalLights.data(); }
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

};
