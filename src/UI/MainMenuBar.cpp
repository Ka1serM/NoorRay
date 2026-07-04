#include "MainMenuBar.h"
#include "imgui.h"
#include "Scene/LightInstance.h"
#include "Scene/MeshInstance.h"
#include "Camera/CameraInstance.h"
#include "Camera/ThinLensCamera.h"
#include "CUDA/rstd/Allocator.h"
#include <SDL3/SDL.h>
#include <iostream>
#include "Log.h"
#include "Scene/SceneImporter.h"
#include <memory>

#include "ImGuiManager.h"

MainMenuBar::MainMenuBar(std::string name, Context& context, Scene& scene, ImGuiManager& imGuiManager)
    : ImGuiComponent(std::move(name)), scene(scene), context(context), imGuiManager(imGuiManager)
{}

void MainMenuBar::renderUi() {
    if (ImGui::BeginMainMenuBar()) {
        renderFileMenu();
        renderAddMenu();
        renderViewMenu();
        ImGui::EndMainMenuBar();
    }
}

void MainMenuBar::renderAddMenu() const {
    if (ImGui::BeginMenu("Add")) {
        if (ImGui::BeginMenu("Primitives")) {
            if (ImGui::MenuItem("Cube")) {
                const uint32_t cube = scene.add(MeshAsset::CreateCube(scene, "Cube", {}));
                auto instance = std::make_unique<MeshInstance>(scene, "Cube Instance", cube, Transform(vec3(0, 0, 0)));
                scene.setActiveObjectId(scene.add(std::move(instance)));
            }
            if (ImGui::MenuItem("Plane")) {
                const uint32_t plane = scene.add(MeshAsset::CreatePlane(scene, "Plane", {}));
                auto instance = std::make_unique<MeshInstance>(scene, "Plane Instance", plane, Transform(vec3(0, 0, 0)));
                scene.setActiveObjectId(scene.add(std::move(instance)));
            }
            if (ImGui::MenuItem("Sphere")) {
                const uint32_t sphere = scene.add(MeshAsset::CreateSphere(scene, "Sphere", {}, 24, 48));
                auto instance = std::make_unique<MeshInstance>(scene, "Sphere Instance", sphere, Transform(vec3(0, 0, 0)));
                scene.setActiveObjectId(scene.add(std::move(instance)));
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Lights")) {
            if (ImGui::MenuItem("Directional Light")) {
                auto light = std::make_unique<LightInstance>(scene, "Directional Light",
                    Transform(vec3(0, 0, 0)), LightInstance::TypeDirectional);
                scene.setActiveObjectId(scene.add(std::move(light)));
            }
            if (ImGui::MenuItem("Point Light")) {
                auto light = std::make_unique<LightInstance>(scene, "Point Light",
                    Transform(vec3(0, 0, 0)), LightInstance::TypePoint);
                scene.setActiveObjectId(scene.add(std::move(light)));
            }
            if (ImGui::MenuItem("Spot Light")) {
                auto light = std::make_unique<LightInstance>(scene, "Spot Light",
                    Transform(vec3(0, 0, 0)), LightInstance::TypeSpot);
                scene.setActiveObjectId(scene.add(std::move(light)));
            }
            if (ImGui::MenuItem("Rect Light")) {
                auto light = std::make_unique<LightInstance>(scene, "Rect Light",
                    Transform(vec3(0, 0, 0)), LightInstance::TypeRect);
                scene.setActiveObjectId(scene.add(std::move(light)));
            }
            ImGui::EndMenu();
        }
        if (ImGui::MenuItem("Camera")) {
            nr::rstd::allocator<ThinLensCamera> allocator;
            ThinLensCamera* thinLens = allocator.allocate(1);
            allocator.construct(thinLens);
            auto camera = std::make_unique<CameraInstance>(
                scene, "Camera", Transform(vec3(0.f, 0.f, 5.f)), Camera(thinLens));
            scene.setActiveObjectId(scene.add(std::move(camera)));
        }
        ImGui::EndMenu();
    }
}

void MainMenuBar::handleFileImport(const std::string& filePath, const FileType type) const
{
    if (filePath.empty())
        return;

    try {
        switch (type) {
            case FileType::OBJ:
                SceneImporter::ImportObjScene(scene, filePath);
                break;
            case FileType::GLTF:
                SceneImporter::ImportGltfScene(scene, filePath);
                break;
            case FileType::TEXTURE:
                scene.add(Texture(context, filePath));
                break;
            case FileType::NRSCENE:
                SceneImporter::ImportJsonScene(scene, filePath);
                break;
            case FileType::PLY:
                SceneImporter::ImportPlyScene(scene, filePath);
                break;
            default:
                break;
        }
    } catch (const std::exception& e) {
       LOG_ERROR("Import failed: " << e.what());
    }
}

void MainMenuBar::renderFileMenu() {
    // Check if a dialog is open and has returned a result.
    if (openDialog && openDialog->ready(0)) {

        const auto& selection = openDialog->result();
        if (!selection.empty())
            handleFileImport(selection[0], pendingFileType);

        // Reset the unique_ptr to close the dialog and reset the state.
        openDialog.reset();
        pendingFileType = FileType::NONE;
    }

    if (ImGui::BeginMenu("File"))
    {
        if (ImGui::BeginMenu("Import")) {
            // Disable menu items if a dialog is currently running.
            ImGui::BeginDisabled(static_cast<bool>(openDialog));

            if (ImGui::MenuItem("Bitmap Texture")) {
                openDialog = std::make_unique<pfd::open_file>("Import Texture", ".", std::vector<std::string>{"Image Files", "*.png *.jpg *.jpeg *.bmp *.tga *.psd *.gif *.hdr *.pic", "All Files", "*"});
                pendingFileType = FileType::TEXTURE;
            }

            if (ImGui::MenuItem("Wavefront .obj")) {
                openDialog = std::make_unique<pfd::open_file>("Import OBJ Model", ".", std::vector<std::string>{"OBJ Files", "*.obj", "All Files", "*"});
                pendingFileType = FileType::OBJ;
            }

            if (ImGui::MenuItem("Khronos .gltf")) {
                openDialog = std::make_unique<pfd::open_file>("Import GLTF", ".", std::vector<std::string>{"GLTF Files", "*.gltf *.glb", "All Files", "*"});
                pendingFileType = FileType::GLTF;
            }

            if (ImGui::MenuItem("NoorRay Scene (.nrscene)")) {
                openDialog = std::make_unique<pfd::open_file>("Import Scene", ".", std::vector<std::string>{"NoorRay Scene", "*.nrscene"});
                pendingFileType = FileType::NRSCENE;
            }

            if (ImGui::MenuItem("3D Gaussians (.ply)")) {
                openDialog = std::make_unique<pfd::open_file>("Import 3DGS Point Cloud", ".", std::vector<std::string>{"PLY Files", "*.ply", "All Files", "*"});
                pendingFileType = FileType::PLY;
            }

            ImGui::EndDisabled();
            ImGui::EndMenu(); // Correctly ends the "Import" menu
        }

        ImGui::Separator();

        if (ImGui::MenuItem("Quit", "Ctrl+Q")) {
            SDL_Event quitEvent;
            quitEvent.type = SDL_EVENT_QUIT;
            SDL_PushEvent(&quitEvent);
        }

        ImGui::EndMenu();
    }
}

void MainMenuBar::renderViewMenu() const
{
    if (ImGui::BeginMenu("View")) {
        if (ImGui::BeginMenu("Theme")) {
            const bool isDarkTheme = (imGuiManager.GetCurrentTheme() == ImGuiManager::Theme::Dark);

            if (ImGui::MenuItem("Dark", nullptr, isDarkTheme))
                imGuiManager.SetTheme(ImGuiManager::Theme::Dark);

            if (ImGui::MenuItem("Light", nullptr, !isDarkTheme))
                imGuiManager.SetTheme(ImGuiManager::Theme::Light);

            ImGui::EndMenu(); // Ends "Theme"
        }

        ImGui::EndMenu(); // Ends "View"
    }
}
