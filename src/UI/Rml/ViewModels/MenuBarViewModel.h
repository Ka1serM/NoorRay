#pragma once
#include "Scene/Scene.h"
#include "ViewModelBase.h"
#include "Scene/SceneImporter.h"
#include "portable-file-dialogs.h"
#include "Mesh/MeshAsset.h"
#include "Scene/MeshInstance.h"

#include <cstdio>
#include <string>
#include <vector>
#include <memory>

enum class FileType { NONE, OBJ, GLTF, TEXTURE };

class MenuBarViewModel : public ViewModelBase {
public:
    MenuBarViewModel(Scene& scene, Rml::Context* ctx, const Rml::String& name);

    // Called every frame to process file dialogs
    void Update();

private:
    Scene& scene;
    std::unique_ptr<pfd::open_file> fileDialog;
    FileType pendingFileType = FileType::NONE;

    // File importing
    void ImportFile(FileType type);
    void ImportFileInternal(const std::string& filePath, FileType type);

    // File Menu
    void OpenScene();
    void ImportObj();
    void ImportGltf();
    void ImportTexture();

    // Add primitives
    void AddCube();
    void AddPlane();
    void AddSphere();

    void AddSphereLight();
    void AddRectLight();
    void AddDiskLight();

    void AddPrimitive(std::shared_ptr<MeshAsset> mesh, const std::string& instanceName);

    // Edit menu
    void Undo();
    void Redo();
    void Cut();
    void Copy();
    void Paste();

    // Help menu
    void About();
    void Quit();
};