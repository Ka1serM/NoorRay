#pragma once

#include "Scene/Scene.h"
#include "Mesh/Assets/Mesh.h"
#include "Scene/SceneObject.h"

class Scene;
class Mesh;

class MeshInstance : public SceneObject {
    Mesh* mesh{};
public:
    MeshInstance(Scene& scene, const std::string& name, Mesh* mesh, const Transform& transf);
    MeshInstance(const MeshInstance& other);


    std::unique_ptr<SceneObject> clone() const override;

    uint32_t getMeshIndex() const { return mesh->getMeshIndex(); }
    Mesh& getMesh() { return *mesh; }
    const Mesh& getMesh() const { return *mesh; }
    Mesh* getMeshPtr() const { return mesh; }
    bool hasMesh() const { return mesh != nullptr; }
    void onTransformUpdated() override;
};
