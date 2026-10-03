#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <unordered_map>
#include <string>
#include <vector>
#include "Materials/MaterialX/MaterialXFwd.h"
#include "Scene/Handle.h"
#include "Shared/RenderSettings.h"
#include "Materials/Material.h"
#include "Mesh/Assets/Mesh.h"
#include "Shared/Light.h"
#include "Texture/Texture.h"
#include "Shared/Math.h"

#include <glm/mat4x4.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "Environment/Environment.h"

class SceneObject;
class MeshInstance;
class CameraInstance;
class LightInstance;
class VolumeInstance;
enum DirtyFlag : uint8_t {
    TLAS         = 1 << 0,
    Meshes       = 1 << 1,
    Textures     = 1 << 2,
    Accumulation = 1 << 3,
    EnvironmentCdf = 1 << 4,
    Lights       = 1 << 5,
    CameraState  = 1 << 6,
    Materials    = 1 << 7,
};

// Receives a scene's changes as they are made, on the thread making them.
class SceneListener {
public:
    virtual ~SceneListener() = default;
    virtual void onHierarchyChanged() {}
    virtual void onActiveObjectChanged() {}
    virtual void onActiveCameraChanged() {}
    virtual void onTexturesChanged() {}
    // The object's own transform changed; a parent's move does not count.
    virtual void onObjectTransformChanged(SceneObjectHandle object) {}
};

class Scene {
    friend class LightInstance;
    friend class SceneObject;
    friend class MeshInstance;
    friend class VolumeInstance;
    friend class Mesh;

    std::deque<Texture> textures;
    uint64_t textureRevision_{1};
    uint64_t materialRevision_{1};
    std::deque<Material> materials;
    std::deque<Mesh> meshes;

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

    std::unordered_map<std::string, Mesh*> meshesByPath_;
    std::unordered_map<std::string, Texture*> texturesByKey_;

    // Mesh instances that have geometry, densely, in the order the renderer
    // numbers them for the TLAS and Cryptomatte. Removal moves the last slot
    // into the hole, so adding, moving and removing each cost O(1).
    std::vector<MeshInstance*> meshInstanceSlots_;
    // What changed since the renderer publishing this scene last took it,
    // each entry listed once.
    std::vector<uint32_t> changedMeshInstanceSlots_;
    std::vector<bool> meshInstanceSlotListed_;
    std::vector<Mesh*> changedMeshes_;
    std::vector<bool> meshListed_;
    std::vector<uint32_t> changedMaterials_;
    std::vector<bool> materialListed_;
    // Materials the MaterialX runtime has yet to compile, each listed once.
    std::vector<uint32_t> materialsToCompile_;
    std::vector<bool> materialCompileListed_;
    std::size_t surfaceMaterialCount_ = 0;
    std::size_t compiledSurfaceMaterialCount_ = 0;
    // Materials only ever gain programs, so the scan for the first pending
    // one resumes where it stopped.
    std::size_t pendingMaterialCursor_ = 0;
    // Light objects by light type, parallel to the typed light records.
    std::array<std::vector<LightInstance*>, 4> lightObjects_;
    // Objects with a billboard, by the slot that is also the viewport's record
    // index. Removal moves the last slot into the hole.
    std::vector<SceneObject*> billboardObjects_;
    // Billboard slots rewritten since the viewport last took them, each listed once.
    std::vector<uint32_t> changedBillboardSlots_;
    std::vector<bool> billboardSlotListed_;
    std::vector<VolumeInstance*> volumeObjects_;
    std::vector<uint32_t> changedVolumeSlots_;
    std::vector<bool> volumeSlotListed_;
    uint64_t volumeStructureRevision_{1};
    // Typed light records rewritten since the session last took them, by
    // light type, each listed once.
    std::array<std::vector<uint32_t>, 4> changedLights_;
    std::array<std::vector<bool>, 4> lightListed_;
    // Every root, and objects that were roots since getRootObjects() last ran;
    // that call drops the stale entries, so each object is dropped once.
    mutable std::vector<SceneObjectHandle> rootCandidates_;
    uint64_t clearEpoch_{};

    std::shared_ptr<CameraInstance> viewportCamera;
    std::weak_ptr<CameraInstance> activeCamera;
    uint64_t activeCameraRevision{};
    // Monotonic version of the object hierarchy and row-visible object data.
    // UI trees use this to avoid walking large scenes on every frame.
    uint64_t hierarchyRevision{1};
    uint64_t lightRevision{1};
    SceneObjectHandle activeObject;
    uint8_t dirtyFlags = 0;
    std::array<uint64_t, 8> changeRevisions_{};

    std::vector<SceneListener*> listeners_;

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
    bool remove(SceneObject* objToRemove);
    void reparent(SceneObject* objectToMove, SceneObject* newParent);
    void notifyGeometryChanged();
    void notifyMaterialChanged() { ++materialRevision_; }
    void addMeshInstanceSlot(MeshInstance& instance);
    void removeMeshInstanceSlot(MeshInstance& instance);
    void markMeshInstanceChanged(uint32_t slot);
    void markLightChanged(int lightType, uint32_t lightIndex);
    void attachOverlays(SceneObject& object);
    void detachOverlays(SceneObject& object);
    void addBillboardSlot(SceneObject& object);
    void removeBillboardSlot(SceneObject& object);
    void markBillboardChanged(uint32_t slot);
    void billboardChanged(SceneObject& object);
    void addVolumeSlot(VolumeInstance& volume);
    void removeVolumeSlot(VolumeInstance& volume);
    void markVolumeChanged(uint32_t slot);
    void markMeshChanged(Mesh& mesh);
    void markMaterialChanged(uint32_t materialIndex);
    void markMaterialForCompile(uint32_t materialIndex);
    void addRootCandidate(SceneObject& object) const;
    void notifyHierarchyChanged();
    void notifyTexturesChanged();
    void notifyObjectTransformChanged(const SceneObject& object);
    void assignActiveObject(SceneObjectHandle handle);
public:
    Scene();
    ~Scene();

    // The listener must stay alive until it is removed.
    void addListener(SceneListener& listener) { listeners_.push_back(&listener); }
    void removeListener(SceneListener& listener) { std::erase(listeners_, &listener); }

    // Marks that CPU-side scene edits must be published before the next GPU
    // snapshot. The session coalesces many UI edits into one wait at the
    // publication boundary.
    void synchronizeBeforeMutation();
    // Releases backend allocations without changing scene contents.
    void releaseGpuResources();
    // Returns true if a GPU sync is needed before the next render frame.
    // The caller (render thread) should sync its stream when this is true.
    bool consumeGpuSync() { return gpuSyncPending_.exchange(false); }

    // Object lifetime
    // With preserveViewportState, remove all scene content while retaining
    // render settings, the current camera view and the environment (including
    // its HDRI). Importers use the default full reset before loading a file.
    void clear(bool preserveViewportState = false, bool preserveEnvironmentTexture = true);
    // With a parent, the object's transform is relative to that parent.
    SceneObjectHandle add(std::unique_ptr<SceneObject> sceneObject, SceneObjectHandle parent = {});
    bool removeObject(SceneObjectHandle handle);
    bool replaceObject(SceneObject* oldObject, std::unique_ptr<SceneObject> newObject);

    Mesh* add(Mesh mesh, bool reuseExisting = true);
    Material* add(Material material);
    // Adds a native material. Importers that only carry the simple
    // BasicMaterial record lower it to a canonical MaterialX document first
    // (nr::materialx::documentFromBasicMaterial) and pass the resulting document
    // here, so every material compiles through the same MaterialX shader
    // pipeline as authored graphs. A null document is allowed: it means an
    // un-authored slot whose document is lowered on demand from the default
    // material. `flags` are MaterialFlag* bits (Shared/Material.h).
    Material* addMaterial(MaterialX::DocumentPtr material, uint32_t flags = 0);
    // Replaces the document of an existing material slot in place, leaving the
    // slot's compiled program untouched. Used by live-editing paths (Hydra)
    // that republish a document while a replacement compiles in the background.
    // A null document clears the slot's authored graph (the default MaterialX
    // material is lowered on demand).
    void updateMaterialDocument(Material* material, MaterialX::DocumentPtr document);
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
    // mesh/material/texture resource the originals reference -- the
    // clones are new SceneObjects/MeshInstances, not new GPU uploads. Used by
    // paste() and by SceneImporter's file-level import cache to instance a
    // previously imported file without re-parsing or re-uploading it.
    std::shared_ptr<SceneObject> cloneHierarchy(const SceneObject* source);

    // Lookup
    bool isValid(SceneObjectHandle handle) const;
    SceneObject* getObject(SceneObjectHandle handle) const { return findObjectPtr(handle).get(); }
    std::shared_ptr<SceneObject> getObjectPtr(SceneObjectHandle handle) const { return findObjectPtr(handle); }
    const std::vector<std::shared_ptr<SceneObject>>& getSceneObjects() const { return sceneObjects; }
    uint64_t getHierarchyRevision() const { return hierarchyRevision; }
    std::vector<std::shared_ptr<SceneObject>> getRootObjects() const;
    // The mesh instances with geometry, by the slot the renderer draws them
    // in; a slot is also the instance's Cryptomatte id.
    const std::vector<MeshInstance*>& getMeshInstanceSlots() const { return meshInstanceSlots_; }
    // What the renderer must republish since its last call: instance slots
    // that were added, refilled, transformed or given other materials, meshes
    // whose geometry changed, and materials that were added or recompiled. One
    // renderer publishes a scene, so taking them empties the lists.
    std::vector<uint32_t> takeChangedMeshInstanceSlots();
    std::vector<Mesh*> takeChangedMeshes();
    std::vector<uint32_t> takeChangedMaterials();
    // Materials added or invalidated since the last call, for the one
    // material compiler of this scene.
    std::vector<uint32_t> takeMaterialsToCompile();
    // Advances whenever clear() empties the scene, which invalidates every
    // index a consumer holds.
    uint64_t getClearEpoch() const { return clearEpoch_; }
    std::vector<const LightInstance*> getLightObjects() const;
    const LightInstance& getLightObject(int lightType, uint32_t lightIndex) const;
    // Record indices by LightInstance light type.
    using LightIndices = std::array<std::vector<uint32_t>, 4>;
    // The typed light records rewritten since the last call, so consumers
    // patch what changed rather than revisit every light. Taking them
    // empties the lists.
    LightIndices takeChangedLights();
    // The objects the viewport draws an icon for, by slot.
    const std::vector<SceneObject*>& getBillboardObjects() const { return billboardObjects_; }
    // The billboard slots rewritten since the last call, so the viewport
    // patches what changed. Taking them empties the list.
    std::vector<uint32_t> takeChangedBillboardSlots();
    const std::vector<VolumeInstance*>& getVolumeObjects() const { return volumeObjects_; }
    uint64_t getVolumeStructureRevision() const { return volumeStructureRevision_; }
    std::vector<uint32_t> takeChangedVolumeSlots();
    uint32_t getActiveCryptomatteId() const;
    // Inverse of getActiveCryptomatteId: the object a rendered id belongs to,
    // or nullptr for the background and stale ids.
    SceneObject* findCryptomatteObject(uint32_t id) const;
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
    void setMaterialProgram(std::size_t materialIndex, MaterialShaderProgram shaderProgram);
    // Surface materials, and those among them that have a program; kept as
    // counts because the UI and viewport ask every frame.
    std::size_t surfaceMaterialCount() const { return surfaceMaterialCount_; }
    std::size_t compiledSurfaceMaterialCount() const { return compiledSurfaceMaterialCount_; }
    // The first surface material still without a program, or null.
    const Material* firstPendingSurfaceMaterial();
    uint32_t getMaterialIndex(const Material* material) const;
    const Material& getMaterial(const Material* material) const { return *material; }
    Material& getMaterial(Material* material) { return *material; }
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
    // Selects a live scene object. An invalid/empty handle clears selection;
    // a stale handle is rejected so UI models cannot publish dangling state.
    bool setActiveObject(SceneObjectHandle handle);
    void clearActiveObject() { assignActiveObject({}); }
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
    // Each renderer keeps its own cursor. Clearing legacy dirty flags cannot
    // hide a mutation from another consumer of the same scene.
    using ChangeState = std::array<uint64_t, 8>;
    ChangeState getChangeState() const { return changeRevisions_; }
    uint8_t changesSince(const ChangeState& previous) const {
        uint8_t changes = 0;
        for (unsigned i = 0; i < changeRevisions_.size(); ++i)
            if (previous[i] != changeRevisions_[i]) changes |= (1u << i);
        return changes;
    }
    void setDirtyFlag(DirtyFlag flag) {
        dirtyFlags |= flag;
        for (unsigned i = 0; i < changeRevisions_.size(); ++i)
            if (static_cast<uint8_t>(flag) & (1u << i)) ++changeRevisions_[i];
        if (flag == Lights)
            ++lightRevision;
    }
    void clearDirtyFlag(DirtyFlag flag) { dirtyFlags &= ~flag; }
    bool isDirty(DirtyFlag flag) const { return (dirtyFlags & flag) != 0; }
    bool isAnyDirty() const { return dirtyFlags & (TLAS | Meshes | Textures
        | EnvironmentCdf | Lights | CameraState | Materials); }
    void clearDirtyFlags() { dirtyFlags = 0; }
    void clearAccumulationDirtyFlag() { dirtyFlags &= ~Accumulation; }

};
