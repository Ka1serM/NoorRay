#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "DiffRenderer/Core/Image.h"
#include "NoorRHI/NoorRHI.h"

namespace DiffRenderer::Scenes {

//! A pinhole camera in the OpenCV convention: x right, y down, z forward.
struct Camera {
    //! The rows of the world-to-camera transform.
    std::array<NoorRHI::float4, 3> worldToCamera{{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}}};
    //! fx, fy, cx and cy in pixels.
    NoorRHI::float4 intrinsics{};
    std::uint32_t width = 0;
    std::uint32_t height = 0;

    NoorRHI::float3 getPosition() const;
};

struct View {
    std::string name;
    Camera camera;
    //! Already composited on the background of the dataset when the source has alpha.
    Core::Image image;
    //! The alpha of the source in [0, 1] for each pixel, or empty.
    std::vector<float> alpha;
};

struct PointCloud {
    std::vector<NoorRHI::float3> positions;
    std::vector<NoorRHI::float3> colors;
};

enum class Kind : std::uint8_t { Scene, Object, Volume };

//! A scalar volume in [0, 1] that fills the cube [-0.5, 0.5]^3, x fastest.
struct Volume {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t depth = 0;
    std::vector<float> values;
    //! The RGBA transfer function that rendered the targets, evenly spaced over [0, 1].
    std::vector<NoorRHI::float4> transferFunction;
    //! The ray marching step in world units that rendered the targets.
    float stepSize = 0.0f;
};

struct Dataset {
    std::string id;
    std::string name;
    Kind kind = Kind::Scene;
    std::vector<View> train;
    std::vector<View> test;
    //! The start points for the methods that need them: SfM points or random points.
    PointCloud points;
    NoorRHI::float3 background{0, 0, 0};
    //! The centre and radius that the 3DGS paper calls the scene extent.
    NoorRHI::float3 center{0, 0, 0};
    float extent = 1.0f;
    //! The known volume of a volume scenario, for the methods that may use it.
    std::optional<Volume> volume;
};

struct DatasetInfo {
    std::string_view id;
    std::string_view name;
    Kind kind;
};

struct LoadOptions {
    //! Divides the image size of every view; 1 keeps the standard size of the benchmark.
    std::uint32_t downscale = 1;
    //! Keeps only every n-th train view, for quick tests.
    std::uint32_t trainStride = 1;
    //! Keeps only every n-th test view, for quick evaluations.
    std::uint32_t testStride = 1;
};

//! The datasets that the data folder can hold, whether downloaded or not.
std::span<const DatasetInfo> getDatasetInfos();
bool isAvailable(std::string_view id, const std::filesystem::path& dataRoot);
//! Loads a COLMAP model with pinhole cameras; every 8th image is a test view, as in the 3DGS benchmark. The image files may be smaller than the model's cameras.
[[nodiscard]] std::expected<Dataset, std::string> loadColmap(const std::filesystem::path& model, const std::filesystem::path& imageFolder,
                                                             const LoadOptions& options);
[[nodiscard]] std::expected<Dataset, std::string> loadDataset(std::string_view id, const std::filesystem::path& dataRoot, const LoadOptions& options);

}
