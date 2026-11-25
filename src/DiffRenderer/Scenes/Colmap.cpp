#include <algorithm>
#include <format>
#include <fstream>
#include <unordered_map>

#include "DiffRenderer/Scenes/Loading.h"

namespace DiffRenderer::Scenes {

namespace {

constexpr std::size_t testInterval = 8;

class BinaryReader {
public:
    explicit BinaryReader(const std::filesystem::path& path) : stream(path, std::ios::binary) {}

    bool isGood() const { return static_cast<bool>(stream); }

    template <class T>
    T read() {
        T value{};
        stream.read(reinterpret_cast<char*>(&value), sizeof(T));
        return value;
    }

    std::string readText() {
        std::string text;
        std::getline(stream, text, '\0');
        return text;
    }

    void skip(const std::streamoff size) { stream.seekg(size, std::ios::cur); }

private:
    std::ifstream stream;
};

struct ColmapCamera {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    NoorRHI::float4 intrinsics{};
};

struct ColmapImage {
    std::string name;
    Matrix3 rotation{};
    Vector3 translation{};
    std::uint32_t cameraId = 0;
};

//! COLMAP model ids: 0 SIMPLE_PINHOLE, 1 PINHOLE, 2 SIMPLE_RADIAL, 3 RADIAL; the radial terms are dropped as in 3DGS.
std::expected<ColmapCamera, std::string> readCameraModel(BinaryReader& reader, const std::int32_t model, const std::uint32_t width,
                                                         const std::uint32_t height) {
    constexpr std::array<std::size_t, 4> parameterCounts{3, 4, 4, 5};
    if (model < 0 || model >= static_cast<std::int32_t>(parameterCounts.size()))
        return std::unexpected(std::format("the COLMAP camera model {} is not a pinhole model", model));
    std::array<double, 5> parameters{};
    for (std::size_t index = 0; index < parameterCounts[static_cast<std::size_t>(model)]; ++index)
        parameters[index] = reader.read<double>();
    const bool separateFocals = model == 1;
    const double focalY = separateFocals ? parameters[1] : parameters[0];
    const double centerX = separateFocals ? parameters[2] : parameters[1];
    const double centerY = separateFocals ? parameters[3] : parameters[2];
    return ColmapCamera{
        width, height, {static_cast<float>(parameters[0]), static_cast<float>(focalY), static_cast<float>(centerX), static_cast<float>(centerY)}};
}

std::expected<std::unordered_map<std::uint32_t, ColmapCamera>, std::string> readCameras(const std::filesystem::path& path) {
    BinaryReader reader(path);
    if (!reader.isGood())
        return std::unexpected(std::format("{} does not open", path.string()));
    std::unordered_map<std::uint32_t, ColmapCamera> cameras;
    const auto count = reader.read<std::uint64_t>();
    for (std::uint64_t index = 0; index < count; ++index) {
        const auto id = reader.read<std::uint32_t>();
        const auto model = reader.read<std::int32_t>();
        const auto width = static_cast<std::uint32_t>(reader.read<std::uint64_t>());
        const auto height = static_cast<std::uint32_t>(reader.read<std::uint64_t>());
        std::expected<ColmapCamera, std::string> camera = readCameraModel(reader, model, width, height);
        if (!camera)
            return std::unexpected(std::move(camera.error()));
        cameras.emplace(id, *camera);
    }
    if (!reader.isGood())
        return std::unexpected(std::format("{} is truncated", path.string()));
    return cameras;
}

Matrix3 toRotation(const double w, const double x, const double y, const double z) {
    return {{{1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y},
             {2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x},
             {2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y}}};
}

std::expected<std::vector<ColmapImage>, std::string> readImages(const std::filesystem::path& path) {
    BinaryReader reader(path);
    if (!reader.isGood())
        return std::unexpected(std::format("{} does not open", path.string()));
    std::vector<ColmapImage> images(reader.read<std::uint64_t>());
    constexpr std::streamoff observationSize = 2 * sizeof(double) + sizeof(std::int64_t);
    for (ColmapImage& image : images) {
        reader.skip(sizeof(std::uint32_t));
        std::array<double, 7> pose{};
        for (double& value : pose)
            value = reader.read<double>();
        image.rotation = toRotation(pose[0], pose[1], pose[2], pose[3]);
        image.translation = {pose[4], pose[5], pose[6]};
        image.cameraId = reader.read<std::uint32_t>();
        image.name = reader.readText();
        reader.skip(static_cast<std::streamoff>(reader.read<std::uint64_t>()) * observationSize);
    }
    if (!reader.isGood())
        return std::unexpected(std::format("{} is truncated", path.string()));
    std::ranges::sort(images, {}, &ColmapImage::name);
    return images;
}

std::expected<PointCloud, std::string> readPoints(const std::filesystem::path& path) {
    BinaryReader reader(path);
    if (!reader.isGood())
        return std::unexpected(std::format("{} does not open", path.string()));
    const auto count = reader.read<std::uint64_t>();
    PointCloud points;
    points.positions.reserve(count);
    points.colors.reserve(count);
    constexpr float colorScale = 1.0f / 255.0f;
    for (std::uint64_t index = 0; index < count; ++index) {
        reader.skip(sizeof(std::uint64_t));
        const auto x = reader.read<double>();
        const auto y = reader.read<double>();
        const auto z = reader.read<double>();
        const auto color = reader.read<std::array<std::uint8_t, 3>>();
        reader.skip(sizeof(double));
        reader.skip(static_cast<std::streamoff>(reader.read<std::uint64_t>()) * 2 * sizeof(std::int32_t));
        points.positions.push_back({static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)});
        points.colors.push_back({color[0] * colorScale, color[1] * colorScale, color[2] * colorScale});
    }
    if (!reader.isGood())
        return std::unexpected(std::format("{} is truncated", path.string()));
    return points;
}

}

std::expected<Dataset, std::string> loadColmap(const std::filesystem::path& model, const std::filesystem::path& imageFolder, const LoadOptions& options) {
    if (options.downscale == 0 || options.trainStride == 0 || options.testStride == 0)
        return std::unexpected("the downscale and the strides must be at least 1");
    std::expected<std::unordered_map<std::uint32_t, ColmapCamera>, std::string> cameras = readCameras(model / "cameras.bin");
    if (!cameras)
        return std::unexpected(std::move(cameras.error()));
    std::expected<std::vector<ColmapImage>, std::string> images = readImages(model / "images.bin");
    if (!images)
        return std::unexpected(std::move(images.error()));
    std::expected<PointCloud, std::string> points = readPoints(model / "points3D.bin");
    if (!points)
        return std::unexpected(std::move(points.error()));
    DatasetSource source;
    for (std::size_t index = 0; index < images->size(); ++index) {
        const ColmapImage& image = (*images)[index];
        const auto camera = cameras->find(image.cameraId);
        if (camera == cameras->end())
            return std::unexpected(std::format("{}: the image {} has the unknown camera {}", model.string(), image.name, image.cameraId));
        const ColmapCamera& intrinsics = camera->second;
        std::vector<ViewSource>& split = index % testInterval == 0 ? source.test : source.train;
        split.push_back(
            {image.name, makeCamera(image.rotation, image.translation, intrinsics.intrinsics, intrinsics.width, intrinsics.height), imageFolder / image.name});
    }
    Dataset dataset;
    dataset.points = std::move(*points);
    setBounds(dataset, source.train);
    if (std::expected<void, std::string> loaded = loadViews(dataset, source, options); !loaded)
        return std::unexpected(std::move(loaded.error()));
    return dataset;
}

std::expected<Dataset, std::string> loadMip360(const std::filesystem::path& folder, const std::uint32_t imageScale, const LoadOptions& options) {
    return loadColmap(folder / "sparse/0", folder / std::format("images_{}", imageScale), options);
}

}
