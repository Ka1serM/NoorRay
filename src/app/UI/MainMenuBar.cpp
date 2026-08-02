#include "MainMenuBar.h"
#include "imgui.h"
#include "Scene/Objects/LightInstance.h"
#include "Scene/Objects/MeshInstance.h"
#include "Rendering/Camera/CameraInstance.h"
#include <SDL3/SDL.h>
#include <iostream>
#include "Log.h"
#include "Scene/Import/SceneImporter.h"
#include "Scene/Import/SceneReader.h"
#include "Scene/Import/SceneUsd.h"
#include "Scene/Import/SceneWriter.h"
#include <algorithm>
#include <cctype>
#include <memory>
#include <filesystem>
#include <fstream>

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

void MainMenuBar::renderAddMenu() {
    if (ImGui::BeginMenu("Add")) {
        if (ImGui::BeginMenu("Primitives")) {
            if (ImGui::MenuItem("Cube")) {
                const MeshAssetRef cube = scene.add(MeshAsset::CreateCube(scene, "Cube", {}));
                auto instance = std::make_unique<MeshInstance>(scene, "Cube Instance", cube, Transform(vec3(0, 0, 0)));
                scene.setActiveObject(scene.add(std::move(instance)));
            }
            if (ImGui::MenuItem("Plane")) {
                const MeshAssetRef plane = scene.add(MeshAsset::CreatePlane(scene, "Plane", {}));
                auto instance = std::make_unique<MeshInstance>(scene, "Plane Instance", plane, Transform(vec3(0, 0, 0)));
                scene.setActiveObject(scene.add(std::move(instance)));
            }
            if (ImGui::MenuItem("Sphere")) {
                const MeshAssetRef sphere = scene.add(MeshAsset::CreateSphere(scene, "Sphere", {}, 24, 48));
                auto instance = std::make_unique<MeshInstance>(scene, "Sphere Instance", sphere, Transform(vec3(0, 0, 0)));
                scene.setActiveObject(scene.add(std::move(instance)));
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Lights")) {
            if (ImGui::MenuItem("Directional Light")) {
                auto light = std::make_unique<LightInstance>(scene, "Directional Light",
                    Transform(vec3(0, 0, 0)), LightInstance::TypeDirectional);
                scene.setActiveObject(scene.add(std::move(light)));
            }
            if (ImGui::MenuItem("Point Light")) {
                auto light = std::make_unique<LightInstance>(scene, "Point Light",
                    Transform(vec3(0, 0, 0)), LightInstance::TypePoint);
                scene.setActiveObject(scene.add(std::move(light)));
            }
            if (ImGui::MenuItem("Spot Light")) {
                auto light = std::make_unique<LightInstance>(scene, "Spot Light",
                    Transform(vec3(0, 0, 0)), LightInstance::TypeSpot);
                scene.setActiveObject(scene.add(std::move(light)));
            }
            if (ImGui::MenuItem("Rect Light")) {
                auto light = std::make_unique<LightInstance>(scene, "Rect Light",
                    Transform(vec3(0, 0, 0)), LightInstance::TypeRect);
                scene.setActiveObject(scene.add(std::move(light)));
            }
            ImGui::EndMenu();
        }
        if (ImGui::MenuItem("Camera")) {
            auto camera = std::make_unique<ThinLensCamera>();
            auto instance = std::make_unique<CameraInstance>(
                std::move(camera), "Camera", Transform(vec3(0.f, 0.f, 5.f)));
            scene.setActiveObject(scene.add(std::move(instance)));
        }
        ImGui::EndMenu();
    }
}

void MainMenuBar::handleMaterialXImport(const std::string& filePath)
{
    if (filePath.empty())
        return;
    try {
        const std::filesystem::path absolute = std::filesystem::absolute(filePath);
        if (!std::filesystem::is_regular_file(absolute))
            throw std::runtime_error("MaterialX file not found: " + filePath);
        const MaterialRef material = scene.addMaterial(MaterialX::DocumentPtr{});
        auto& paths = scene.getMaterialXSourcePaths();
        auto& documents = scene.getMaterialXDocuments();
        if (paths.size() <= material.index()) paths.resize(material.index() + 1);
        if (documents.size() <= material.index()) documents.resize(material.index() + 1);
        // The path is the source of truth for a disk-backed material: its XML
        // is read at compile time (import), never retained in memory.
        paths[material.index()] = absolute.string();
        documents[material.index()] = nullptr;
        scene.invalidateMaterial(material.handle());
    } catch (const std::exception& error) {
        LOG_ERROR("MaterialX import failed: " << error.what());
    }
}

void MainMenuBar::handleFileImport(const std::string& filePath)
{
    if (filePath.empty())
        return;

    const auto extension = std::filesystem::path(filePath).extension().string();
    if (extension == ".mtx" || extension == ".mtlx") {
        handleMaterialXImport(filePath);
        return;
    }

    if (extension == ".pbrt") {
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
        std::string extension = std::filesystem::path(filePath).extension().string();
        std::ranges::transform(extension, extension.begin(), [](const unsigned char value) {
            return static_cast<char>(std::tolower(value));
        });
        currentScenePath = nr::sceneio::isUsdFile(filePath) || extension == ".pbrt"
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
        path.replace_extension(".usd");

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
                std::vector<std::string>{"Supported Scenes", "*.usd *.usda *.usdc *.nrscene *.pbrt", "USD Scene", "*.usd *.usda *.usdc", "Legacy NoorRay Scene", "*.nrscene", "PBRT Scene", "*.pbrt", "All Files", "*"});
        }

        if (ImGui::MenuItem("Save", nullptr, false, !currentScenePath.empty()))
            saveScene(currentScenePath);

        if (ImGui::MenuItem("Save As...")) {
            std::filesystem::path defaultPath = currentScenePath.empty()
                ? std::filesystem::path("scene.usd")
                : std::filesystem::path(currentScenePath);
            // Save As starts with the USD filter selected. Do not carry a
            // legacy .nrscene suffix into that filter when the current scene
            // was opened from the old JSON format.
            if (!nr::sceneio::isUsdFile(defaultPath.string()))
                defaultPath.replace_extension(".usd");
            sceneSaveDialog = std::make_unique<pfd::save_file>(
                "Save Scene As",
                defaultPath.string(),
                std::vector<std::string>{"USD Scene", "*.usd *.usda *.usdc", "PBRT Scene", "*.pbrt", "Legacy NoorRay Scene", "*.nrscene", "All Files", "*"},
                pfd::opt::force_overwrite);
        }

        ImGui::Separator();

        if (ImGui::MenuItem("Import...")) {
            openDialog = std::make_unique<pfd::open_file>(
                "Import Asset",
                ".",
                std::vector<std::string>{
                    "Supported Assets",
                    "*.obj *.gltf *.glb *.mtx *.mtlx *.ply *.compressed.ply *.splat *.ksplat *.spz *.sog *.png *.jpg *.jpeg *.bmp *.tga *.psd *.gif *.hdr *.pic",
                    "Gaussian Splats",
                    "*.ply *.compressed.ply *.splat *.ksplat *.spz *.sog",
                    "Meshes",
                    "*.obj *.gltf *.glb",
                    "MaterialX",
                    "*.mtx *.mtlx",
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
