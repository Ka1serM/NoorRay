#include "MainMenuBar.h"
#include "imgui.h"
#include "Scene/LightInstance.h"
#include "Scene/MeshInstance.h"
#include "Camera/CameraInstance.h"
#include <SDL3/SDL.h>
#include <iostream>
#include "Log.h"
#include "Scene/SceneImporter.h"
#include "Scene/SceneReader.h"
#include "Scene/SceneWriter.h"
#include <memory>
#include <filesystem>

#include "ImGuiManager.h"

MainMenuBar::MainMenuBar(std::string name, Context& context, Scene& scene,
    ImGuiManager& imGuiManager, std::string currentScenePath)
    : ImGuiComponent(std::move(name)), scene(scene), context(context),
      imGuiManager(imGuiManager), currentScenePath(std::move(currentScenePath))
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
            auto camera = std::make_unique<ThinLensCamera>();
            auto instance = std::make_unique<CameraInstance>(
                std::move(camera), "Camera", Transform(vec3(0.f, 0.f, 5.f)));
            scene.setActiveObjectId(scene.add(std::move(instance)));
        }
        ImGui::EndMenu();
    }
}

void MainMenuBar::handleFileImport(const std::string& filePath)
{
    if (filePath.empty())
        return;

    if (std::filesystem::path(filePath).extension() == ".pbrt") {
        openScene(filePath);
        return;
    }

    try {
        SceneImporter::ImportFile(scene, filePath);
    } catch (const std::exception& e) {
       LOG_ERROR("Import failed: " << e.what());
    }
}

void MainMenuBar::openScene(const std::string& filePath)
{
    if (filePath.empty())
        return;

    try {
        scene.load(filePath);
        currentScenePath = std::filesystem::path(filePath).extension() == ".nrscene"
            ? filePath : std::string{};
    } catch (const std::exception& e) {
        LOG_ERROR("Open scene failed: " << e.what());
    }
}

void MainMenuBar::saveScene(const std::string& filePath)
{
    if (filePath.empty())
        return;

    std::filesystem::path path(filePath);
    if (path.extension().empty())
        path.replace_extension(".nrscene");

    try {
        SceneWriter::Write(scene, path.string());
        currentScenePath = path.string();
    } catch (const std::exception& e) {
        LOG_ERROR("Save scene failed: " << e.what());
    }
}

void MainMenuBar::renderFileMenu() {
    // Check if a dialog is open and has returned a result.
    if (openDialog && openDialog->ready(0)) {

        const auto& selection = openDialog->result();
        if (!selection.empty())
            handleFileImport(selection[0]);

        // Reset the unique_ptr to close the dialog and reset the state.
        openDialog.reset();
    }

    if (sceneOpenDialog && sceneOpenDialog->ready(0)) {
        const auto& selection = sceneOpenDialog->result();
        if (!selection.empty())
            openScene(selection[0]);
        sceneOpenDialog.reset();
    }

    if (sceneSaveDialog && sceneSaveDialog->ready(0)) {
        saveScene(sceneSaveDialog->result());
        sceneSaveDialog.reset();
    }

    if (ImGui::BeginMenu("File"))
    {
        const bool dialogOpen = static_cast<bool>(openDialog)
            || static_cast<bool>(sceneOpenDialog)
            || static_cast<bool>(sceneSaveDialog);

        ImGui::BeginDisabled(dialogOpen);
        if (ImGui::MenuItem("Open...")) {
            sceneOpenDialog = std::make_unique<pfd::open_file>(
                "Open Scene",
                ".",
                std::vector<std::string>{"Supported Scenes", "*.nrscene *.pbrt", "PBRT Scene", "*.pbrt", "All Files", "*"});
        }

        if (ImGui::MenuItem("Save", nullptr, false, !currentScenePath.empty()))
            saveScene(currentScenePath);

        if (ImGui::MenuItem("Save As...")) {
            const std::string defaultPath = currentScenePath.empty() ? "scene.nrscene" : currentScenePath;
            sceneSaveDialog = std::make_unique<pfd::save_file>(
                "Save Scene As",
                defaultPath,
                std::vector<std::string>{"NoorRay Scene", "*.nrscene", "All Files", "*"},
                pfd::opt::force_overwrite);
        }

        ImGui::Separator();

        if (ImGui::MenuItem("Import...")) {
            openDialog = std::make_unique<pfd::open_file>(
                "Import Asset",
                ".",
                std::vector<std::string>{
                    "Supported Assets",
                    "*.obj *.gltf *.glb *.ply *.compressed.ply *.splat *.ksplat *.spz *.sog *.png *.jpg *.jpeg *.bmp *.tga *.psd *.gif *.hdr *.pic",
                    "Gaussian Splats",
                    "*.ply *.compressed.ply *.splat *.ksplat *.spz *.sog",
                    "Meshes",
                    "*.obj *.gltf *.glb",
                    "Images",
                    "*.png *.jpg *.jpeg *.bmp *.tga *.psd *.gif *.hdr *.pic",
                    "All Files",
                    "*"});
        }
        ImGui::EndDisabled();

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
