#include "MainMenuBar.h"
#include "imgui.h"
#include "portable-file-dialogs.h"
#include "Utils.h"
#include "Scene/MeshInstance.h"
#include <SDL3/SDL.h>
#include <iostream>
#include <future>

MainMenuBar::MainMenuBar(std::string name, Context& context, Scene& scene) 
    : ImGuiComponent(std::move(name)), scene(scene), context(context) {}

void MainMenuBar::renderUi() {
    if (ImGui::BeginMainMenuBar()) {
        renderFileMenu();
        renderAddMenu();
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
                int instanceIndex = scene.add(std::move(instance));
                scene.setActiveObjectIndex(instanceIndex);
            }
            if (ImGui::MenuItem("Plane")) {
                auto plane = MeshAsset::CreatePlane(scene, "Plane", {});
                scene.add(plane);
                auto instance = std::make_unique<MeshInstance>(scene, "Plane Instance", plane, Transform(vec3(0, 0, 0)));
                int instanceIndex = scene.add(std::move(instance));
                scene.setActiveObjectIndex(instanceIndex);
            }
            if (ImGui::MenuItem("Sphere")) {
                auto sphere = MeshAsset::CreateSphere(scene, "Sphere", {}, 24, 48);
                scene.add(sphere);
                auto instance = std::make_unique<MeshInstance>(scene, "Sphere Instance", sphere, Transform(vec3(0, 0, 0)));
                int instanceIndex = scene.add(std::move(instance));
                scene.setActiveObjectIndex(instanceIndex);
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
                int instanceIndex = scene.add(std::move(instance));
                scene.setActiveObjectIndex(instanceIndex);
            }
            if (ImGui::MenuItem("Rect Light")) {
                Material material{};
                material.emission = vec3(1, 1, 1);
                material.emissionStrength = 10.0f;
                auto plane = MeshAsset::CreatePlane(scene, "RectLight", material);
                scene.add(plane);
                auto instance = std::make_unique<MeshInstance>(scene, "RectLight Instance", plane, Transform(vec3(0, 0, 0)));
                int instanceIndex = scene.add(std::move(instance));
                scene.setActiveObjectIndex(instanceIndex);
            }
            if (ImGui::MenuItem("Disk Light")) {
                Material material{};
                material.emission = vec3(1, 1, 1);
                material.emissionStrength = 10.0f;
                auto disk = MeshAsset::CreateDisk(scene, "DiskLight", material, 48);
                scene.add(disk);
                auto instance = std::make_unique<MeshInstance>(scene, "DiskLight Instance", disk, Transform(vec3(0, 0, 0)));
                int instanceIndex = scene.add(std::move(instance));
                scene.setActiveObjectIndex(instanceIndex);
            }
            ImGui::EndMenu();
        }
        ImGui::EndMenu();
    }
}

// Function to handle the import logic once a file is selected
void MainMenuBar::handleFileImport(const std::string& filePath, FileType type) const
{
    if (filePath.empty()) {
        return;
    }

    try {
        switch (type) {
            case FileType::OBJ: {
                std::vector<Vertex> vertices;
                std::vector<uint32_t> indices;
                std::vector<Face> faces;
                std::vector<Material> materials;
                Utils::loadObj(scene, filePath, vertices, indices, faces, materials);
                auto meshAsset = std::make_shared<MeshAsset>(scene, filePath, std::move(vertices), std::move(indices), std::move(faces), std::move(materials));
                scene.add(meshAsset);
                auto instance = std::make_unique<MeshInstance>(scene, Utils::nameFromPath(meshAsset->getPath()) + " Instance", meshAsset, Transform{});
                int instanceIndex = scene.add(std::move(instance));
                scene.setActiveObjectIndex(instanceIndex);
                break;
            }
            case FileType::CRTSCENE: {
                std::vector<Vertex> vertices;
                std::vector<uint32_t> indices;
                std::vector<Face> faces;
                std::vector<Material> materials;
                Utils::loadCrtScene(scene, filePath, vertices, indices, faces, materials);
                auto meshAsset = std::make_shared<MeshAsset>(scene, filePath, std::move(vertices), std::move(indices), std::move(faces), std::move(materials));
                scene.add(meshAsset);
                auto instance = std::make_unique<MeshInstance>(scene, Utils::nameFromPath(meshAsset->getPath()) + " Instance", meshAsset, Transform{});
                int instanceIndex = scene.add(std::move(instance));
                scene.setActiveObjectIndex(instanceIndex);
                break;
            }
            case FileType::TEXTURE: {
                scene.add(Texture(context, filePath));
                break;
            }
            default:
                break;
        }
    } catch (const std::exception& e) {
        std::cerr << "Import failed: " << e.what() << std::endl;
    }
}

void MainMenuBar::renderFileMenu() {
    // Check if any async file dialog has completed
    if (openFuture.valid() && openFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        const auto selection = openFuture.get();
        if (!selection.empty()) {
            handleFileImport(selection[0], pendingFileType);
        }
    }

    if (ImGui::BeginMenu("File")) {
        ImGui::Separator();

        if (ImGui::BeginMenu("Import")) {
            // Disable menu items if a dialog is already open
            ImGui::BeginDisabled(openFuture.valid() && openFuture.wait_for(std::chrono::seconds(0)) != std::future_status::ready);
            
            if (ImGui::MenuItem("Wavefront .obj")) {
                openFuture = std::async(std::launch::async, [] {
                    return pfd::open_file("Import OBJ Model", ".", { "OBJ Files", "*.obj", "All Files", "*" }).result();
                });
                pendingFileType = FileType::OBJ;
            }

            if (ImGui::MenuItem("Chaos Camp .crtscene")) {
                openFuture = std::async(std::launch::async, [] {
                    return pfd::open_file("Import CrtScene", ".", { "CRT Scene Files", "*.crtscene", "All Files", "*" }).result();
                });
                pendingFileType = FileType::CRTSCENE;
            }

            if (ImGui::MenuItem("Bitmap Texture")) {
                openFuture = std::async(std::launch::async, [] {
                    return pfd::open_file("Import Texture", ".", { "Image Files", "*.png *.jpg *.jpeg *.bmp *.tga *.psd *.gif *.hdr *.pic", "All Files", "*" }).result();
                });
                pendingFileType = FileType::TEXTURE;
            }
            
            ImGui::EndDisabled();
            ImGui::EndMenu();
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