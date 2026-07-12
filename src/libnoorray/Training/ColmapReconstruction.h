#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

namespace noorray
{
enum class ReconstructionMapper
{
    Global,
    Incremental,
};

struct ColmapTrainingView
{
    std::filesystem::path imagePath;
    glm::mat4 cameraToWorld{1.0f};
    float fx{}, fy{}, cx{}, cy{};
    uint32_t width{};
    uint32_t height{};
};

struct ColmapReconstruction
{
    std::vector<glm::vec3> points;
    std::vector<glm::vec3> colors;
    std::vector<ColmapTrainingView> views;
};

ColmapReconstruction reconstructImagesWithColmap(
    const std::filesystem::path& imageDirectory,
    const std::filesystem::path& workspaceDirectory,
    ReconstructionMapper mapper = ReconstructionMapper::Global);
}
