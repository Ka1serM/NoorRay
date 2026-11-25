#include "DiffRenderer/Scenes/Dataset.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <format>
#include <thread>

#include "DiffRenderer/Core/ImageFile.h"
#include "DiffRenderer/Core/Random.h"
#include "DiffRenderer/Scenes/Loading.h"

namespace DiffRenderer::Scenes {

namespace {

constexpr std::array datasetInfos{
    DatasetInfo{"nerf-chair", "NeRF-Synthetic chair", Kind::Object},    DatasetInfo{"nerf-drums", "NeRF-Synthetic drums", Kind::Object},
    DatasetInfo{"nerf-ficus", "NeRF-Synthetic ficus", Kind::Object},    DatasetInfo{"nerf-hotdog", "NeRF-Synthetic hotdog", Kind::Object},
    DatasetInfo{"nerf-lego", "NeRF-Synthetic lego", Kind::Object},      DatasetInfo{"nerf-materials", "NeRF-Synthetic materials", Kind::Object},
    DatasetInfo{"nerf-mic", "NeRF-Synthetic mic", Kind::Object},        DatasetInfo{"nerf-ship", "NeRF-Synthetic ship", Kind::Object},
    DatasetInfo{"mip360-garden", "Mip-NeRF 360 garden", Kind::Scene},   DatasetInfo{"mip360-bonsai", "Mip-NeRF 360 bonsai", Kind::Scene},
    DatasetInfo{"mip360-counter", "Mip-NeRF 360 counter", Kind::Scene}, DatasetInfo{"volume-phantom", "Head phantom (96^3)", Kind::Volume},
    DatasetInfo{"volume-brain", "MSD BrainTumour FLAIR", Kind::Volume},
};

constexpr std::string_view nerfPrefix = "nerf-";
constexpr std::string_view mipPrefix = "mip360-";

std::filesystem::path getNerfFolder(const std::filesystem::path& dataRoot, const std::string_view id) {
    return dataRoot / "nerf_synthetic" / id.substr(nerfPrefix.size());
}

std::filesystem::path getMipFolder(const std::filesystem::path& dataRoot, const std::string_view id) {
    return dataRoot / "mip360" / id.substr(mipPrefix.size());
}

std::uint32_t getMipImageScale(const std::string_view id) {
    return id == "mip360-garden" ? 4 : 2;
}

std::filesystem::path getBrainFolder(const std::filesystem::path& dataRoot) {
    return dataRoot / "msd";
}

const DatasetInfo* findInfo(const std::string_view id) {
    const auto found = std::ranges::find(datasetInfos, id, &DatasetInfo::id);
    return found == datasetInfos.end() ? nullptr : &*found;
}

void composite(Core::ImageFile& file, const NoorRHI::float3 background) {
    const std::span<float> pixels = file.image.getPixels();
    const std::array<float, 3> fill{background.x, background.y, background.z};
    for (std::size_t pixel = 0; pixel < file.alpha.size(); ++pixel)
        for (std::size_t channel = 0; channel < 3; ++channel) {
            float& value = pixels[pixel * 3 + channel];
            value = value * file.alpha[pixel] + fill[channel] * (1.0f - file.alpha[pixel]);
        }
}

std::vector<float> downscaleChannels(const std::span<const float> values, const std::uint32_t width, const std::uint32_t channels, const std::uint32_t factor,
                                     const std::uint32_t targetWidth, const std::uint32_t targetHeight) {
    std::vector<float> result(std::size_t{targetWidth} * targetHeight * channels);
    const float weight = 1.0f / static_cast<float>(factor * factor);
    for (std::uint32_t y = 0; y < targetHeight; ++y)
        for (std::uint32_t sourceY = y * factor; sourceY < (y + 1) * factor; ++sourceY)
            for (std::uint32_t x = 0; x < targetWidth; ++x)
                for (std::uint32_t sourceX = x * factor; sourceX < (x + 1) * factor; ++sourceX)
                    for (std::uint32_t channel = 0; channel < channels; ++channel)
                        result[(std::size_t{y} * targetWidth + x) * channels + channel] +=
                            weight * values[(std::size_t{sourceY} * width + sourceX) * channels + channel];
    return result;
}

void downscale(Core::ImageFile& file, const std::uint32_t factor) {
    if (factor == 1)
        return;
    const std::uint32_t width = file.image.getWidth();
    const std::uint32_t targetWidth = width / factor;
    const std::uint32_t targetHeight = file.image.getHeight() / factor;
    Core::Image image(targetWidth, targetHeight);
    image.setEncoding(file.image.getEncoding());
    std::ranges::copy(downscaleChannels(file.image.getPixels(), width, 3, factor, targetWidth, targetHeight), image.getPixels().begin());
    file.image = std::move(image);
    if (!file.alpha.empty())
        file.alpha = downscaleChannels(file.alpha, width, 1, factor, targetWidth, targetHeight);
}

Camera fitCamera(Camera camera, const std::uint32_t fileWidth, const std::uint32_t fileHeight, const std::uint32_t factor) {
    const float scaleX = static_cast<float>(fileWidth) / static_cast<float>(camera.width * factor);
    const float scaleY = static_cast<float>(fileHeight) / static_cast<float>(camera.height * factor);
    camera.intrinsics = {camera.intrinsics.x * scaleX, camera.intrinsics.y * scaleY, camera.intrinsics.z * scaleX, camera.intrinsics.w * scaleY};
    camera.width = fileWidth / factor;
    camera.height = fileHeight / factor;
    return camera;
}

std::expected<std::vector<View>, std::string> loadStridedViews(const std::vector<ViewSource>& sources, const std::uint32_t stride, const Dataset& dataset,
                                                               const std::uint32_t factor) {
    std::vector<View> views((sources.size() + stride - 1) / stride);
    std::vector<std::string> errors(views.size());
    parallelFor(views.size(), [&](const std::size_t index) {
        const ViewSource& source = sources[index * stride];
        std::expected<Core::ImageFile, std::string> file = Core::loadImageFile(source.file);
        if (!file) {
            errors[index] = std::move(file.error());
            return;
        }
        if (!file->alpha.empty())
            composite(*file, dataset.background);
        const Camera camera = fitCamera(source.camera, file->image.getWidth(), file->image.getHeight(), factor);
        downscale(*file, factor);
        views[index] = View{source.name, camera, std::move(file->image), std::move(file->alpha)};
    });
    for (std::string& error : errors)
        if (!error.empty())
            return std::unexpected(std::move(error));
    return views;
}

}

NoorRHI::float3 Camera::getPosition() const {
    const auto& [first, second, third] = worldToCamera;
    return {-(first.x * first.w + second.x * second.w + third.x * third.w), -(first.y * first.w + second.y * second.w + third.y * third.w),
            -(first.z * first.w + second.z * second.w + third.z * third.w)};
}

Camera makeCamera(const Matrix3& rotation, const Vector3& translation, const NoorRHI::float4& intrinsics, const std::uint32_t width,
                  const std::uint32_t height) {
    Camera camera{.intrinsics = intrinsics, .width = width, .height = height};
    for (std::size_t row = 0; row < 3; ++row)
        camera.worldToCamera[row] = {static_cast<float>(rotation[row][0]), static_cast<float>(rotation[row][1]), static_cast<float>(rotation[row][2]),
                                     static_cast<float>(translation[row])};
    return camera;
}

void setBounds(Dataset& dataset, const std::vector<ViewSource>& train) {
    Vector3 mean{};
    for (const ViewSource& view : train) {
        const NoorRHI::float3 position = view.camera.getPosition();
        mean = {mean[0] + position.x, mean[1] + position.y, mean[2] + position.z};
    }
    const double count = static_cast<double>(train.size());
    mean = {mean[0] / count, mean[1] / count, mean[2] / count};
    double largest = 0.0;
    for (const ViewSource& view : train) {
        const NoorRHI::float3 position = view.camera.getPosition();
        largest = std::max(largest, std::hypot(position.x - mean[0], position.y - mean[1], position.z - mean[2]));
    }
    dataset.center = {static_cast<float>(mean[0]), static_cast<float>(mean[1]), static_cast<float>(mean[2])};
    dataset.extent = static_cast<float>(1.1 * largest);
}

std::expected<void, std::string> loadViews(Dataset& dataset, const DatasetSource& source, const LoadOptions& options) {
    std::expected<std::vector<View>, std::string> train = loadStridedViews(source.train, options.trainStride, dataset, options.downscale);
    if (!train)
        return std::unexpected(std::move(train.error()));
    std::expected<std::vector<View>, std::string> test = loadStridedViews(source.test, options.testStride, dataset, options.downscale);
    if (!test)
        return std::unexpected(std::move(test.error()));
    dataset.train = std::move(*train);
    dataset.test = std::move(*test);
    return {};
}

PointCloud makeRandomPoints(const std::uint32_t count, const float low, const float high, const std::uint64_t seed) {
    Core::Random random(seed);
    PointCloud points;
    points.positions.resize(count);
    points.colors.resize(count);
    for (std::uint32_t index = 0; index < count; ++index) {
        points.positions[index] = {random.uniform(low, high), random.uniform(low, high), random.uniform(low, high)};
        points.colors[index] = {random.uniform(0.0f, 1.0f), random.uniform(0.0f, 1.0f), random.uniform(0.0f, 1.0f)};
    }
    return points;
}

void parallelFor(const std::size_t count, const std::function<void(std::size_t)>& task) {
    std::atomic<std::size_t> next = 0;
    const std::size_t workerCount = std::min<std::size_t>(std::max(1u, std::thread::hardware_concurrency()), count);
    std::vector<std::jthread> workers;
    workers.reserve(workerCount);
    for (std::size_t worker = 0; worker < workerCount; ++worker)
        workers.emplace_back([&] {
            for (std::size_t index = next++; index < count; index = next++)
                task(index);
        });
}

namespace {

std::expected<Dataset, std::string> loadKind(const std::string_view id, const std::filesystem::path& dataRoot, const LoadOptions& options) {
    const std::filesystem::path cache = dataRoot / "cache" / id;
    if (id.starts_with(nerfPrefix))
        return loadNerfSynthetic(id.substr(nerfPrefix.size()), getNerfFolder(dataRoot, id), options);
    if (id.starts_with(mipPrefix))
        return loadMip360(getMipFolder(dataRoot, id), getMipImageScale(id), options);
    if (id == "volume-phantom")
        return loadPhantom(cache, options);
    return loadBrain(getBrainFolder(dataRoot), cache, options);
}

}

std::span<const DatasetInfo> getDatasetInfos() {
    return datasetInfos;
}

bool isAvailable(const std::string_view id, const std::filesystem::path& dataRoot) {
    if (!findInfo(id))
        return false;
    if (id.starts_with(nerfPrefix)) {
        const std::filesystem::path folder = getNerfFolder(dataRoot, id);
        return std::filesystem::exists(folder / "transforms_train.json") && std::filesystem::exists(folder / "transforms_test.json")
               && std::filesystem::exists(folder / "train/r_99.png") && std::filesystem::exists(folder / "test/r_199.png");
    }
    if (id.starts_with(mipPrefix)) {
        const std::filesystem::path folder = getMipFolder(dataRoot, id);
        return std::filesystem::exists(folder / "sparse/0/images.bin") && std::filesystem::exists(folder / std::format("images_{}", getMipImageScale(id)));
    }
    if (id == "volume-brain")
        return std::filesystem::exists(getBrainFolder(dataRoot) / "imagesTr/BRATS_001.nii.gz");
    return true;
}

std::expected<Dataset, std::string> loadDataset(const std::string_view id, const std::filesystem::path& dataRoot, const LoadOptions& options) {
    const DatasetInfo* info = findInfo(id);
    if (!info)
        return std::unexpected(std::format("unknown dataset {}", id));
    if (!isAvailable(id, dataRoot))
        return std::unexpected(std::format("the dataset {} is not in {}", id, dataRoot.string()));
    if (options.downscale == 0 || options.trainStride == 0 || options.testStride == 0)
        return std::unexpected("the downscale and the strides must be at least 1");
    std::expected<Dataset, std::string> dataset = loadKind(id, dataRoot, options);
    if (!dataset)
        return dataset;
    dataset->id = info->id;
    dataset->name = info->name;
    dataset->kind = info->kind;
    return dataset;
}

}
