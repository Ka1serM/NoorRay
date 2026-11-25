#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#include "DiffRenderer/Scenes/Dataset.h"

namespace DiffRenderer::Scenes {

using Vector3 = std::array<double, 3>;
//! Row-major.
using Matrix3 = std::array<Vector3, 3>;

struct ViewSource {
    std::string name;
    //! Its size is the size that the intrinsics refer to; the loader rescales them to the size of the file.
    Camera camera;
    std::filesystem::path file;
};

struct DatasetSource {
    std::vector<ViewSource> train;
    std::vector<ViewSource> test;
};

Camera makeCamera(const Matrix3& rotation, const Vector3& translation, const NoorRHI::float4& intrinsics, std::uint32_t width, std::uint32_t height);
//! Sets the centre and extent from the train cameras as 3DGS does (getNerfppNorm).
void setBounds(Dataset& dataset, const std::vector<ViewSource>& train);
//! Loads the images of every stride-th view, composites them on the background and downscales them.
[[nodiscard]] std::expected<void, std::string> loadViews(Dataset& dataset, const DatasetSource& source, const LoadOptions& options);
PointCloud makeRandomPoints(std::uint32_t count, float low, float high, std::uint64_t seed);
//! Runs the task for every index in [0, count) on all cores.
void parallelFor(std::size_t count, const std::function<void(std::size_t)>& task);

[[nodiscard]] std::expected<Dataset, std::string> loadNerfSynthetic(std::string_view object, const std::filesystem::path& folder, const LoadOptions& options);
[[nodiscard]] std::expected<Dataset, std::string> loadMip360(const std::filesystem::path& folder, std::uint32_t imageScale, const LoadOptions& options);
[[nodiscard]] std::expected<Dataset, std::string> loadPhantom(const std::filesystem::path& cache, const LoadOptions& options);
[[nodiscard]] std::expected<Dataset, std::string> loadBrain(const std::filesystem::path& folder, const std::filesystem::path& cache,
                                                            const LoadOptions& options);

}
