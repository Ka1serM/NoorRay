#include "MainMenuBar.h"
#include "imgui.h"
#include "Scene/MeshInstance.h"
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
                auto cube = MeshAsset::CreateCube(scene, "Cube", {});
                scene.add(cube);
                auto instance = std::make_unique<MeshInstance>(scene, "Cube Instance", cube, Transform(vec3(0, 0, 0)));
                scene.setActiveObjectId(scene.add(std::move(instance)));
            }
            if (ImGui::MenuItem("Plane")) {
                auto plane = MeshAsset::CreatePlane(scene, "Plane", {});
                scene.add(plane);
                auto instance = std::make_unique<MeshInstance>(scene, "Plane Instance", plane, Transform(vec3(0, 0, 0)));
                scene.setActiveObjectId(scene.add(std::move(instance)));
            }
            if (ImGui::MenuItem("Sphere")) {
                auto sphere = MeshAsset::CreateSphere(scene, "Sphere", {}, 24, 48);
                scene.add(sphere);
                auto instance = std::make_unique<MeshInstance>(scene, "Sphere Instance", sphere, Transform(vec3(0, 0, 0)));
                scene.setActiveObjectId(scene.add(std::move(instance)));
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Lights")) {
            if (ImGui::MenuItem("Sphere Light")) {
                Material material{};
                material.emission = vec3(1, 1, 1);
                material.emissionStrength = 10.0f;
                auto sphere = MeshAsset::CreateSphere(scene, "SphereLight", material, 24, 48);
                scene.add(sphere);
                auto instance = std::make_unique<MeshInstance>(scene, "SphereLight Instance", sphere, Transform(vec3(0, 0, 0)));
                scene.setActiveObjectId(scene.add(std::move(instance)));
            }
            if (ImGui::MenuItem("Rect Light")) {
                Material material{};
                material.emission = vec3(1, 1, 1);
                material.emissionStrength = 10.0f;
                auto plane = MeshAsset::CreatePlane(scene, "RectLight", material);
                scene.add(plane);
                auto instance = std::make_unique<MeshInstance>(scene, "RectLight Instance", plane, Transform(vec3(0, 0, 0)));
                scene.setActiveObjectId(scene.add(std::move(instance)));
            }
            if (ImGui::MenuItem("Disk Light")) {
                Material material{};
                material.emission = vec3(1, 1, 1);
                material.emissionStrength = 10.0f;
                auto disk = MeshAsset::CreateDisk(scene, "DiskLight", material, 48);
                scene.add(disk);
                auto instance = std::make_unique<MeshInstance>(scene, "DiskLight Instance", disk, Transform(vec3(0, 0, 0)));
                scene.setActiveObjectId(scene.add(std::move(instance)));
            }
            ImGui::EndMenu();
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
            
            if (ImGui::MenuItem("Wavefront .obj")) {
                openDialog = std::make_unique<pfd::open_file>("Import OBJ Model", ".", std::vector<std::string>{"OBJ Files", "*.obj", "All Files", "*"});
                pendingFileType = FileType::OBJ;
            }

            if (ImGui::MenuItem("Khronos .gltf")) {
                openDialog = std::make_unique<pfd::open_file>("Import GLTF", ".", std::vector<std::string>{"GLTF Files", "*.gltf *.glb", "All Files", "*"});
                pendingFileType = FileType::GLTF;
            }

            if (ImGui::MenuItem("Bitmap Texture")) {
                openDialog = std::make_unique<pfd::open_file>("Import Texture", ".", std::vector<std::string>{"Image Files", "*.png *.jpg *.jpeg *.bmp *.tga *.psd *.gif *.hdr *.pic", "All Files", "*"});
                pendingFileType = FileType::TEXTURE;
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
