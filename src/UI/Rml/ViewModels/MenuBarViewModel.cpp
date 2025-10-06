#include "MenuBarViewModel.h"
#include <SDL3/SDL.h>

MenuBarViewModel::MenuBarViewModel(Scene& scene, Rml::Context* ctx, const Rml::String& name)
    : ViewModelBase(ctx, name), scene(scene)
{
    // File Menu
    BindAction("import_scene", this, &MenuBarViewModel::OpenScene);
    BindAction("import_obj", this, &MenuBarViewModel::ImportObj);
    BindAction("import_gltf", this, &MenuBarViewModel::ImportGltf);
    BindAction("import_texture", this, &MenuBarViewModel::ImportTexture);

    // Add Menu
    BindAction("add_cube", this, &MenuBarViewModel::AddCube);
    BindAction("add_plane", this, &MenuBarViewModel::AddPlane);
    BindAction("add_sphere", this, &MenuBarViewModel::AddSphere);

    // Lights Menu
    BindAction("add_sphere_light", this, &MenuBarViewModel::AddSphereLight);
    BindAction("add_rect_light", this, &MenuBarViewModel::AddRectLight);
    BindAction("add_disk_light", this, &MenuBarViewModel::AddDiskLight);

    // Edit Menu
    BindAction("undo", this, &MenuBarViewModel::Undo);
    BindAction("redo", this, &MenuBarViewModel::Redo);
    BindAction("cut", this, &MenuBarViewModel::Cut);
    BindAction("copy", this, &MenuBarViewModel::Copy);
    BindAction("paste", this, &MenuBarViewModel::Paste);

    // Help Menu
    BindAction("about", this, &MenuBarViewModel::About);
    BindAction("quit", this, &MenuBarViewModel::Quit);
}

// Called every frame
void MenuBarViewModel::Update() {
    if (!fileDialog) return;

    if (fileDialog->ready()) {
        auto selection = fileDialog->result();
        if (!selection.empty()) {
            const std::string& filePath = selection[0];
            ImportFileInternal(filePath, pendingFileType);
        }

        fileDialog.reset();
        pendingFileType = FileType::NONE;
    }
}

void MenuBarViewModel::ImportFile(FileType type) {
    if (fileDialog) return; // already open

    std::string title;
    std::vector<std::string> filters;

    switch (type) {
        case FileType::OBJ:
            title = "Import OBJ Model";
            filters = {"OBJ Files", "*.obj", "All Files", "*"};
            break;
        case FileType::GLTF:
            title = "Import GLTF";
            filters = {"GLTF Files", "*.gltf *.glb", "All Files", "*"};
            break;
        case FileType::TEXTURE:
            title = "Import Texture";
            filters = {"Image Files", "*.png *.jpg *.jpeg *.bmp *.tga *.psd *.gif *.hdr *.pic", "All Files", "*"};
            break;
        default: return;
    }

    fileDialog = std::make_unique<pfd::open_file>(title, ".", filters);
    pendingFileType = type;
}

void MenuBarViewModel::ImportFileInternal(const std::string& filePath, FileType type) {
    try {
        switch (type) {
            case FileType::OBJ:
                SceneImporter::ImportObjScene(scene, filePath);
                break;
            case FileType::GLTF:
                SceneImporter::ImportGltfScene(scene, filePath);
                break;
            case FileType::TEXTURE:
                scene.add(Texture(scene.getContext(), filePath));
                break;
            default: break;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Import failed: %s\n", e.what());
    }
}

// File Menu
void MenuBarViewModel::OpenScene() { std::printf("New scene created\n"); }
void MenuBarViewModel::ImportObj() { ImportFile(FileType::OBJ); }
void MenuBarViewModel::ImportGltf() { ImportFile(FileType::GLTF); }
void MenuBarViewModel::ImportTexture() { ImportFile(FileType::TEXTURE); }

// Add primitives
void MenuBarViewModel::AddCube() { AddPrimitive(MeshAsset::CreateCube(scene, "Cube", {}), "Cube Instance"); }
void MenuBarViewModel::AddPlane() { AddPrimitive(MeshAsset::CreatePlane(scene, "Plane", {}), "Plane Instance"); }
void MenuBarViewModel::AddSphere() { AddPrimitive(MeshAsset::CreateSphere(scene, "Sphere", {}, 24, 48), "Sphere Instance"); }

void MenuBarViewModel::AddSphereLight() {
    Material mat{};
    mat.emission = vec3(1,1,1);
    mat.emissionStrength = 10.0f;
    AddPrimitive(MeshAsset::CreateSphere(scene, "SphereLight", mat, 24, 48), "SphereLight Instance");
}

void MenuBarViewModel::AddRectLight() {
    Material mat{};
    mat.emission = vec3(1,1,1);
    mat.emissionStrength = 10.0f;
    AddPrimitive(MeshAsset::CreatePlane(scene, "RectLight", mat), "RectLight Instance");
}

void MenuBarViewModel::AddDiskLight() {
    Material mat{};
    mat.emission = vec3(1,1,1);
    mat.emissionStrength = 10.0f;
    AddPrimitive(MeshAsset::CreateDisk(scene, "DiskLight", mat, 48), "DiskLight Instance");
}

void MenuBarViewModel::AddPrimitive(std::shared_ptr<MeshAsset> mesh, const std::string& instanceName) {
    scene.add(mesh);
    auto instance = std::make_unique<MeshInstance>(scene, instanceName, mesh, Transform(vec3(0,0,0)));
    uint32_t idx = scene.add(std::move(instance));
    scene.setActiveObject(idx);
}

// Edit menu
void MenuBarViewModel::Undo() {}
void MenuBarViewModel::Redo() {}
void MenuBarViewModel::Cut() {}
void MenuBarViewModel::Copy() {}
void MenuBarViewModel::Paste() {}

// Help menu
void MenuBarViewModel::About() {}
void MenuBarViewModel::Quit() {
    SDL_Event quitEvent{};
    quitEvent.type = SDL_EVENT_QUIT;
    SDL_PushEvent(&quitEvent);
}