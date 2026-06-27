#pragma once

#include <memory>
#include <string>
#include <vector>
#include "Scene/Scene.h"
#include "Scene/Inspectable.h"
#include "Scene/SceneTypes.h"

class Scene;

class MeshAsset : public Inspectable
{
public:
    static std::shared_ptr<MeshAsset> CreateCube(Scene& scene, const std::string& name, const Material& material);
    static std::shared_ptr<MeshAsset> CreatePlane(Scene& scene, const std::string& name, const Material& material);
    static std::shared_ptr<MeshAsset> CreateSphere(Scene& scene, const std::string& name,  const Material& material, uint32_t latitudeSegments = 64, uint32_t longitudeSegments = 64);
    static std::shared_ptr<MeshAsset> CreateDisk(Scene& scene, const std::string& name, const Material& material, uint32_t segments = 64);
    
    MeshAsset(Scene& context, std::string  name, const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices, const std::vector<Face>& faces, const std::vector<Material>& materials);

    const std::string& getName() const override { return path; }
    std::string getType() const override { return "Mesh Asset"; }
    void renderUi();
    void updateMaterials();

    // Getters & Setters-
    const std::string& getPath() const { return path; }
    uint32_t getMeshIndex() const;
    void setMeshIndex(uint32_t newIndex);
    
    const std::vector<Vertex>& getVertices() const { return vertices; }
    const std::vector<uint32_t>& getIndices() const { return indices; }
    const std::vector<Face>& getFaces() const { return faces; }
    const std::vector<Material>& getMaterials() const { return materials; }

    // Dirty Flag
    bool isDirty() const { return dirty; }
    void clearDirtyFlag() { dirty = false; }

private:
    Scene& scene;
    std::string path;
    uint32_t index = -1;
    bool dirty = false;

    // CPU-side data
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<Face> faces;
    std::vector<Material> materials;

};
