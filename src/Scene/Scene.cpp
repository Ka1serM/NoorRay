#include "Scene.h"
#include "Mesh/MeshAsset.h"
#include "Scene/MeshInstance.h"
#include "Scene/SceneObject.h"
#include "Camera/PerspectiveCamera.h"

Scene::Scene()
{

}

void Scene::addSceneObject(std::unique_ptr<SceneObject> sceneObject) {
    // If the object is a PerspectiveCamera, save a raw pointer to activeCamera
    if (auto camera = dynamic_cast<PerspectiveCamera*>(sceneObject.get()))
        activeCamera = camera;

    sceneObjects.push_back(std::move(sceneObject));
}


void Scene::addMeshInstance(std::unique_ptr<MeshInstance> meshInstance) {
    sceneObjects.push_back(std::unique_ptr<SceneObject>(std::move(meshInstance).release()));
}

void Scene::addMeshAsset(const std::shared_ptr<MeshAsset> &meshAsset) {
    meshAsset->setMeshIndex(meshAssets.size());
    meshAssets.push_back(meshAsset);
}