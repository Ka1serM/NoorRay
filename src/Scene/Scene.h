#pragma once

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
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

    uint32_t selectionIndex = 0;
    uint8_t dirtyFlags = 0;
};

class SceneObject;
class MeshInstance;
class MeshAsset;
class PerspectiveCamera;
class Buffer;

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

    // Owns all objects
    std::unordered_map<int, std::unique_ptr<SceneObject>> objectsById;
    // Non-owning pointers to top-level objects for hierarchy traversal
    std::vector<SceneObject*> rootObjects;

    std::vector<MeshInstance*> meshInstances;
    PerspectiveCamera* activeCamera = nullptr;
    int activeObjectId = -1;
    uint8_t dirtyFlags = 0;

    SceneObject* copiedObject = nullptr;
    int nextObjectId = 0;

public:
    Scene(Context& context);

    // Object management
    int add(std::unique_ptr<SceneObject> obj);
    int add(std::unique_ptr<SceneObject> obj, int parentId);
    bool remove(SceneObject* obj);
    void reparent(SceneObject* obj, SceneObject* newParent);

    // Copy/paste
    void copy(SceneObject* obj);
    SceneObject* cloneHierarchy(const SceneObject* source);
    void paste();

    // Getters
    PerspectiveCamera* getActiveCamera() const { return activeCamera; }
    const std::vector<SceneObject*>& getRootObjects() const { return rootObjects; }
    const std::vector<MeshInstance*>& getMeshInstances() const { return meshInstances; }
    const std::vector<std::shared_ptr<MeshAsset>>& getMeshAssets() const { return meshAssets; }
    const std::vector<Texture>& getTextures() const { return textures; }
    Context& getContext() const { return context; }
    std::vector<std::string> getTextureNames() const { return textureNames; }
    int getActiveObjectId() const { return activeObjectId; }

    SceneObject* getObject(int id) const {
        if (id < 0) return nullptr;
        auto it = objectsById.find(id);
        return (it != objectsById.end()) ? it->second.get() : nullptr;
    }

    SceneObject* getActiveObject() const {
        return getObject(activeObjectId);
    }

    void setActiveObject(int id);

    // Dirty flags
    void setDirtyFlag(DirtyFlag flag);
    void clearDirtyFlag(DirtyFlag flag) { dirtyFlags &= ~flag; Notify({SceneEvent::DirtyFlagsChanged, 0, dirtyFlags}); }
    bool isDirty(DirtyFlag flag) const { return (dirtyFlags & flag) != 0; }
    bool isAnyDirty() const { return dirtyFlags & (TLAS | Meshes | Textures); }
    void clearDirtyFlags() { dirtyFlags = 0; Notify({SceneEvent::DirtyFlagsChanged, 0, dirtyFlags}); }
    void clearAccumulationDirtyFlag() { dirtyFlags &= ~Accumulation; Notify({SceneEvent::DirtyFlagsChanged, 0, dirtyFlags}); }

    // Assets
    void add(const std::shared_ptr<MeshAsset>& meshAsset);
    void add(Texture&& texture);
    std::shared_ptr<MeshAsset> getMeshAsset(const std::string& name) const;
};

