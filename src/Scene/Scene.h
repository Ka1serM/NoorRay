#pragma once

#include <memory>
#include <vector>
#include <string>
#include "Vulkan/Context.h"
#include "Vulkan/Texture.h"
#include "Shaders/SharedStructs.h"
#include "../UI/Rml/Observable/Observable.h"

struct SceneEvent {
    enum Type {
        HierarchyChanged,
        SelectionChanged,
        DirtyFlagsChanged
    } type;

    // Optional payload
    uint32_t selectionIndex = 0;
    uint8_t dirtyFlags = 0;
};

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

class Scene : public Observable<SceneEvent> {
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
    uint32_t activeObjectIndex = INVALID_INSTANCE;
    uint8_t dirtyFlags = 0;

    SceneObject* copiedObject = nullptr;

public:
    Scene(Context& context);

    // Object management
    uint32_t registerObject(std::unique_ptr<SceneObject> sceneObject);
    uint32_t add(std::unique_ptr<SceneObject> sceneObject);
    void add(const std::shared_ptr<MeshAsset>& meshAsset);
    void add(Texture&& texture);
    bool remove(SceneObject* objToRemove);
    void reparent(SceneObject* objectToMove, SceneObject* newParent);

    // Copy-paste
    void copy(SceneObject* objectToCopy);
    SceneObject* cloneHierarchy(const SceneObject* source);
    void paste();

    // Getters
    PerspectiveCamera* getActiveCamera() const { return activeCamera; }
    const std::vector<std::unique_ptr<SceneObject>>& getSceneObjects() const { return sceneObjects; }
    const std::vector<SceneObject*>& getRootObjects() const { return rootObjects; }
    const std::vector<MeshInstance*>& getMeshInstances() const { return meshInstances; }
    const std::vector<std::shared_ptr<MeshAsset>>& getMeshAssets() const { return meshAssets; }
    const std::vector<Texture>& getTextures() const { return textures; }
    Context& getContext() const { return context; }
    std::vector<std::string> getTextureNames() const { return textureNames; }

    SceneObject* getObject(uint32_t index) const {
        if (index < static_cast<uint32_t>(sceneObjects.size()))
            return sceneObjects[index].get();
        return nullptr;
    }

    // Selection
    void setActiveObjectIndex(uint32_t index);
    void resetActiveObjectIndex() { activeObjectIndex = INVALID_INSTANCE; }
    uint32_t getActiveObjectIndex() const { return activeObjectIndex; }
    SceneObject* getActiveObject() const;

    // Dirty flags
    void setDirtyFlag(DirtyFlag flag);
    void clearDirtyFlag(DirtyFlag flag) { dirtyFlags &= ~flag; Notify({SceneEvent::DirtyFlagsChanged, 0, dirtyFlags}); }
    bool isDirty(DirtyFlag flag) const { return (dirtyFlags & flag) != 0; }
    bool isAnyDirty() const { return dirtyFlags & (TLAS | Meshes | Textures); }
    void clearDirtyFlags() { dirtyFlags = 0; Notify({SceneEvent::DirtyFlagsChanged, 0, dirtyFlags}); }
    void clearAccumulationDirtyFlag() { dirtyFlags &= ~Accumulation; Notify({SceneEvent::DirtyFlagsChanged, 0, dirtyFlags}); }

    // Assets
    std::shared_ptr<MeshAsset> getMeshAsset(const std::string& name) const;
};
