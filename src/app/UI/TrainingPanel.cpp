#include "UI/TrainingPanel.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <functional>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include <imgui.h>
#include <glm/common.hpp>
#include <glm/geometric.hpp>

#include "Mesh/Assets/GaussianAsset.h"
#include "Raytracing/Runtime/Raytracer.h"
#include "Scene/GaussianInstance.h"
#include "Scene/Scene.h"
#include "Training/ColmapReconstruction.h"
#include "Training/GaussianTrainRenderer.h"
#include "portable-file-dialogs.h"
#include "stb_image.h"
#include "stb_image_resize2.h"

TrainingPanel::TrainingPanel(std::string name, Scene& scene, Raytracer& raytracer)
    : ImGuiComponent(std::move(name)), scene(scene), raytracer(raytracer)
{
}

TrainingPanel::~TrainingPanel() = default;

void TrainingPanel::reconstructAndBegin()
{
    if (!std::filesystem::is_directory(imageDirectory))
    {
        status = "Select a valid input image folder first.";
        return;
    }
    status = "Reconstructing cameras and sparse points with COLMAP...";
    try
    {
        const std::filesystem::path images = std::filesystem::canonical(imageDirectory);
        std::filesystem::remove_all(images / ".noorray-colmap");
        const std::filesystem::path workspace = std::filesystem::temp_directory_path()
            / ("noorray-colmap-" + std::to_string(std::hash<std::string>{}(images.string())));
        noorray::ColmapReconstruction reconstruction =
            noorray::reconstructImagesWithColmap(images, workspace, reconstructionMapper);
        if (!scene.getGaussianInstances().empty())
            throw std::runtime_error("Remove the existing Gaussian object before starting an image reconstruction");

        std::vector<Gaussian> gaussians(reconstruction.points.size());
        glm::vec3 boundsMin(std::numeric_limits<float>::max());
        glm::vec3 boundsMax(std::numeric_limits<float>::lowest());
        for (const glm::vec3& point : reconstruction.points)
        {
            boundsMin = glm::min(boundsMin, point);
            boundsMax = glm::max(boundsMax, point);
        }
        const float reconstructionDiagonal = glm::length(boundsMax - boundsMin);
        const float initialScale = reconstruction.points.empty() ? 0.01f
            : std::max(1.0e-5f, 0.5f * reconstructionDiagonal
                / std::cbrt(static_cast<float>(reconstruction.points.size())));
        for (size_t i = 0; i < reconstruction.points.size(); ++i)
        {
            Gaussian& gaussian = gaussians[i];
            gaussian.transform = glm::mat4x3(glm::vec3(initialScale, 0, 0),
                glm::vec3(0, initialScale, 0), glm::vec3(0, 0, initialScale),
                reconstruction.points[i]);
            gaussian.opacity = 0.1f;
            gaussian.setShCoefficient(0,
                (reconstruction.colors[i] - glm::vec3(0.5f)) / SphericalHarmonicsC0);
            gaussian.sphericalHarmonics.count = 1;
        }
        const uint32_t assetIndex = scene.add(GaussianAsset(scene, "COLMAP sparse points", std::move(gaussians)));
        scene.add(std::make_unique<GaussianInstance>(scene, "Trained Gaussians", assetIndex, Transform{}));
        trainRenderer = std::make_unique<GaussianTrainRenderer>(raytracer, scene);
        trainer = std::make_unique<noorray::GaussianTrainer>(*trainRenderer, scene, config);
        for (const auto& view : reconstruction.views)
        {
            int width{}, height{}, channels{};
            unsigned char* source = stbi_load(view.imagePath.string().c_str(), &width, &height, &channels, 3);
            if (!source) throw std::runtime_error("Could not load " + view.imagePath.string());

            constexpr int maxTrainingDimension = 1024;
            const size_t scratchCapacity = raytracer.getScratchCapacity();
            const double dimensionScale = static_cast<double>(maxTrainingDimension)
                / static_cast<double>(std::max(width, height));
            const double capacityScale = scratchCapacity > 0
                ? std::sqrt(static_cast<double>(scratchCapacity)
                    / static_cast<double>(static_cast<size_t>(width) * height))
                : 0.0;
            const double scale = std::min({1.0, dimensionScale, capacityScale});
            if (scale <= 0.0)
            {
                stbi_image_free(source);
                throw std::runtime_error("Raytracing scratch buffers are not initialized for training");
            }
            const int trainingWidth = std::max(1, static_cast<int>(std::floor(width * scale)));
            const int trainingHeight = std::max(1, static_cast<int>(std::floor(height * scale)));
            std::vector<unsigned char> resized;
            const unsigned char* trainingSource = source;
            if (trainingWidth != width || trainingHeight != height)
            {
                resized.resize(static_cast<size_t>(trainingWidth) * trainingHeight * 3);
                if (!stbir_resize_uint8_linear(source, width, height, 0, resized.data(),
                        trainingWidth, trainingHeight, 0, STBIR_RGB))
                {
                    stbi_image_free(source);
                    throw std::runtime_error("Could not resize " + view.imagePath.string());
                }
                trainingSource = resized.data();
            }
            std::vector<float> target(static_cast<size_t>(trainingWidth) * trainingHeight * 3);
            std::transform(trainingSource, trainingSource + target.size(), target.begin(),
                [](unsigned char value) { return static_cast<float>(value) / 255.0f; });
            stbi_image_free(source);
            const float scaleX = static_cast<float>(trainingWidth) / static_cast<float>(view.width);
            const float scaleY = static_cast<float>(trainingHeight) / static_cast<float>(view.height);
            noorray::GaussianTrainingCamera camera{view.cameraToWorld,
                view.fx * scaleX, view.fy * scaleY, view.cx * scaleX, view.cy * scaleY};
            trainer->addView(target.data(), trainingWidth, trainingHeight, camera,
                view.imagePath.filename().string());
        }
        lastStep = {};
        running = true;
        stopped = false;
        scene.setDirtyFlag(TLAS);
        status = "COLMAP reconstruction complete. Training started.";
    }
    catch (const std::exception& error)
    {
        trainer.reset();
        trainRenderer.reset();
        running = false;
        status = error.what();
    }
}

void TrainingPanel::selectImageDirectory()
{
    const std::string selected = pfd::select_folder("Select training images", imageDirectory).result();
    if (!selected.empty()) imageDirectory = selected;
}

void TrainingPanel::exportResult()
{
    if (!trainer) return;
    const std::string path = pfd::save_file("Export trained Gaussians", "trained.ply",
        {"Gaussian PLY", "*.ply", "Gaussian SOG", "*.sog"}).result();
    if (path.empty()) return;
    try
    {
        trainer->exportGaussians(path);
        status = "Exported " + path;
    }
    catch (const std::exception& error)
    {
        status = error.what();
    }
}

bool TrainingPanel::tick()
{
    if (!running || !trainer) return false;
    try
    {
        lastStep = trainer->trainStep();
        scene.setDirtyFlag(Accumulation);
        if (lastStep.iteration >= static_cast<uint64_t>(std::max(1, targetIterations)))
        {
            running = false;
            status = "Training target reached.";
        }
        return true;
    }
    catch (const std::exception& error)
    {
        running = false;
        status = error.what();
        return false;
    }
}

void TrainingPanel::renderUi()
{
    ImGui::Begin(name.c_str());
    std::array<char, 1024> pathBuffer{};
    std::copy_n(imageDirectory.c_str(), std::min(imageDirectory.size(), pathBuffer.size() - 1), pathBuffer.data());
    ImGui::SetNextItemWidth(-42.0f);
    if (ImGui::InputText("##imageDirectory", pathBuffer.data(), pathBuffer.size()))
        imageDirectory = pathBuffer.data();
    ImGui::SameLine();
    if (ImGui::Button("...##browseImages")) selectImageDirectory();
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Select input image folder");
    ImGui::Separator();
    int mapperIndex = reconstructionMapper == noorray::ReconstructionMapper::Global ? 0 : 1;
    constexpr const char* mapperNames[] = {"Global (GLOMAP)", "Incremental"};
    if (ImGui::Combo("Reconstruction", &mapperIndex, mapperNames, 2))
        reconstructionMapper = mapperIndex == 0 ? noorray::ReconstructionMapper::Global
                                                : noorray::ReconstructionMapper::Incremental;
    int spp = static_cast<int>(config.samplesPerPixel);
    if (ImGui::DragInt("Samples per pixel", &spp, 0.1f, 1, 64))
        config.samplesPerPixel = static_cast<uint32_t>(std::max(1, spp));
    ImGui::InputInt("Target iterations", &targetIterations);
    targetIterations = std::max(1, targetIterations);
    ImGui::InputScalar("Seed", ImGuiDataType_U64, &config.seed);
    ImGui::DragFloat("Position learning rate", &config.learningRatePosition, 1.0e-5f, 0.0f, 1.0f, "%.6f");
    ImGui::DragFloat("Scale learning rate", &config.learningRateScale, 1.0e-4f, 0.0f, 1.0f, "%.6f");
    ImGui::DragFloat("Rotation learning rate", &config.learningRateRotation, 1.0e-4f, 0.0f, 1.0f, "%.6f");
    ImGui::DragFloat("Opacity learning rate", &config.learningRateOpacity, 1.0e-3f, 0.0f, 1.0f, "%.6f");
    ImGui::DragFloat("Color learning rate", &config.learningRateColor, 1.0e-4f, 0.0f, 1.0f, "%.6f");

    ImGui::BeginDisabled(trainer || imageDirectory.empty());
    if (ImGui::Button("Start")) reconstructAndBegin();
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::BeginDisabled(!trainer || stopped);
    if (ImGui::Button(running ? "Pause" : "Resume")) running = !running;
    ImGui::SameLine();
    if (ImGui::Button("Stop"))
    {
        running = false;
        stopped = true;
        status = "Training stopped. The current result can be exported.";
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::BeginDisabled(!trainer);
    if (ImGui::Button("Export")) exportResult();
    ImGui::EndDisabled();

    ImGui::Separator();
    ImGui::Text("Views: %zu", trainer ? trainer->viewCount() : 0);
    ImGui::Text("Iteration: %llu / %d", static_cast<unsigned long long>(lastStep.iteration), targetIterations);
    ImGui::Text("Loss: %.7f", lastStep.loss);
    ImGui::Text("Gaussians: %u", trainer ? trainer->gaussianCount() : 0);
    ImGui::TextWrapped("%s", status.c_str());
    ImGui::End();
}
