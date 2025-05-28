#pragma once
#include <memory>
#include <vector>

class MeshAsset;
class MeshInstance;
class SceneObject;
class PerspectiveCamera;

class Scene {

    std::vector<std::shared_ptr<MeshAsset>> meshAssets;

    PerspectiveCamera* activeCamera = nullptr;  // non-owning pointer

    size_t selectedObjectIndex = -1;

public:
    std::vector<std::unique_ptr<SceneObject>> sceneObjects;

    Scene();

    void setActiveCamera(PerspectiveCamera* camera) {
        activeCamera = camera;
    }

    PerspectiveCamera* getActiveCamera() const {
        return activeCamera;
    }

    void addSceneObject(std::unique_ptr<SceneObject> sceneObject);

    void addMeshInstance(std::unique_ptr<MeshInstance> meshInstance);

    void addMeshAsset(const std::shared_ptr<MeshAsset>& meshInstance);
};
