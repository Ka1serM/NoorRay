#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <numbers>

#include "DiffRenderer/Algorithms/VolumeRendering/Phantom.h"
#include "DiffRenderer/Core/ImageFile.h"
#include "DiffRenderer/Scenes/Loading.h"
#include "DiffRenderer/Scenes/Nifti.h"

namespace DiffRenderer::Scenes {

namespace {

constexpr std::uint32_t phantomSize = 96;
constexpr std::uint32_t phantomImageSize = 256;
constexpr std::uint32_t brainImageSize = 512;
constexpr std::uint32_t trainViewCount = 64;
constexpr std::uint32_t testViewCount = 32;
constexpr std::uint32_t transferFunctionSize = 64;
constexpr double extinctionScale = 60.0;
constexpr double cameraDistance = 1.9;
constexpr double tanHalfFov = 0.36397023426620234;
constexpr float stopTransmittance = 1e-4f;
constexpr std::uint32_t startPointCount = 100'000;
constexpr std::uint64_t startPointSeed = 1;

struct Bump {
    double center;
    double width;
    double opacity;
    std::array<double, 3> color;
};

//! A blend of Gaussian bumps; the opacity of a bump is scaled to an extinction per world unit.
std::vector<NoorRHI::float4> makeTransferFunction(const std::span<const Bump> bumps, const double cutoff) {
    std::vector<NoorRHI::float4> entries(transferFunctionSize);
    constexpr double baseWeight = 0.05;
    for (std::uint32_t entry = 0; entry < transferFunctionSize; ++entry) {
        const double value = static_cast<double>(entry) / (transferFunctionSize - 1);
        std::array<double, 3> color{0.5 * baseWeight, 0.5 * baseWeight, 0.5 * baseWeight};
        double opacity = 0.0;
        double weightSum = baseWeight;
        for (const Bump& bump : bumps) {
            const double weight = std::exp(-0.5 * std::pow((value - bump.center) / bump.width, 2.0));
            opacity += bump.opacity * weight;
            for (std::size_t channel = 0; channel < 3; ++channel)
                color[channel] += weight * bump.color[channel];
            weightSum += weight;
        }
        const double extinction = value < cutoff ? 0.0 : std::min(opacity, 1.0) * extinctionScale;
        entries[entry] = {static_cast<float>(color[0] / weightSum), static_cast<float>(color[1] / weightSum), static_cast<float>(color[2] / weightSum),
                          static_cast<float>(extinction)};
    }
    return entries;
}

std::vector<NoorRHI::float4> makePhantomTransferFunction() {
    constexpr std::array bumps{Bump{0.12, 0.03, 0.1, {0.3, 0.5, 0.95}}, Bump{0.22, 0.04, 0.03, {0.95, 0.75, 0.6}}, Bump{0.45, 0.05, 0.04, {0.85, 0.8, 0.75}},
                               Bump{0.74, 0.03, 0.6, {1.0, 0.8, 0.2}}, Bump{0.95, 0.05, 0.06, {1.0, 1.0, 0.95}}};
    return makeTransferFunction(bumps, 0.06);
}

std::vector<NoorRHI::float4> makeBrainTransferFunction() {
    constexpr std::array bumps{Bump{0.25, 0.08, 0.02, {0.55, 0.6, 0.75}}, Bump{0.45, 0.06, 0.04, {0.9, 0.85, 0.8}}, Bump{0.75, 0.06, 0.5, {1.0, 0.55, 0.15}},
                               Bump{0.95, 0.04, 0.9, {1.0, 0.2, 0.15}}};
    return makeTransferFunction(bumps, 0.08);
}

class ReferenceMarcher {
public:
    explicit ReferenceMarcher(const Volume& volume) : volume(volume) {}

    void render(const Camera& camera, Core::Image& image) const {
        const std::span<float> pixels = image.getPixels();
        for (std::uint32_t y = 0; y < camera.height; ++y)
            for (std::uint32_t x = 0; x < camera.width; ++x) {
                const std::array<float, 3> color = trace(camera, x, y);
                std::ranges::copy(color, pixels.begin() + (std::size_t{y} * camera.width + x) * 3);
            }
    }

private:
    float sample(const std::array<float, 3>& position) const {
        const std::array<std::uint32_t, 3> size{volume.width, volume.height, volume.depth};
        std::array<std::int32_t, 3> base{};
        std::array<float, 3> fraction{};
        for (std::size_t axis = 0; axis < 3; ++axis) {
            const float coordinate = (position[axis] + 0.5f) * static_cast<float>(size[axis]) - 0.5f;
            const float floor = std::floor(coordinate);
            base[axis] = static_cast<std::int32_t>(floor);
            fraction[axis] = coordinate - floor;
        }
        float result = 0.0f;
        for (std::uint32_t corner = 0; corner < 8; ++corner) {
            float weight = 1.0f;
            std::array<std::int32_t, 3> voxel{};
            for (std::size_t axis = 0; axis < 3; ++axis) {
                const bool high = (corner >> axis & 1u) != 0;
                voxel[axis] = base[axis] + (high ? 1 : 0);
                weight *= high ? fraction[axis] : 1.0f - fraction[axis];
            }
            if (voxel[0] < 0 || voxel[1] < 0 || voxel[2] < 0 || voxel[0] >= static_cast<std::int32_t>(size[0]) || voxel[1] >= static_cast<std::int32_t>(size[1])
                || voxel[2] >= static_cast<std::int32_t>(size[2]))
                continue;
            result += weight
                      * volume.values[(static_cast<std::size_t>(voxel[2]) * size[1] + static_cast<std::size_t>(voxel[1])) * size[0]
                                      + static_cast<std::size_t>(voxel[0])];
        }
        return result;
    }

    NoorRHI::float4 classify(const float value) const {
        const float position = std::clamp(value, 0.0f, 1.0f) * static_cast<float>(volume.transferFunction.size() - 1);
        const float base = std::min(std::floor(position), static_cast<float>(volume.transferFunction.size() - 2));
        const float fraction = position - base;
        const NoorRHI::float4& low = volume.transferFunction[static_cast<std::size_t>(base)];
        const NoorRHI::float4& high = volume.transferFunction[static_cast<std::size_t>(base) + 1];
        return {low.x + (high.x - low.x) * fraction, low.y + (high.y - low.y) * fraction, low.z + (high.z - low.z) * fraction,
                low.w + (high.w - low.w) * fraction};
    }

    std::array<float, 3> trace(const Camera& camera, const std::uint32_t x, const std::uint32_t y) const {
        const std::array<float, 3> local{(static_cast<float>(x) + 0.5f - camera.intrinsics.z) / camera.intrinsics.x,
                                         (static_cast<float>(y) + 0.5f - camera.intrinsics.w) / camera.intrinsics.y, 1.0f};
        const auto& rows = camera.worldToCamera;
        std::array<float, 3> direction{rows[0].x * local[0] + rows[1].x * local[1] + rows[2].x * local[2],
                                       rows[0].y * local[0] + rows[1].y * local[1] + rows[2].y * local[2],
                                       rows[0].z * local[0] + rows[1].z * local[1] + rows[2].z * local[2]};
        const float length = std::hypot(direction[0], direction[1], direction[2]);
        for (float& component : direction)
            component /= length;
        const NoorRHI::float3 eye = camera.getPosition();
        const std::array<float, 3> origin{eye.x, eye.y, eye.z};
        float entry = 0.0f;
        float exit = std::numeric_limits<float>::infinity();
        for (std::size_t axis = 0; axis < 3; ++axis) {
            const float low = (-0.5f - origin[axis]) / direction[axis];
            const float high = (0.5f - origin[axis]) / direction[axis];
            entry = std::max(entry, std::min(low, high));
            exit = std::min(exit, std::max(low, high));
        }
        std::array<float, 3> color{};
        float transmittance = 1.0f;
        for (std::uint32_t step = 0; transmittance > stopTransmittance; ++step) {
            const float distance = entry + (static_cast<float>(step) + 0.5f) * volume.stepSize;
            if (distance >= exit)
                break;
            const NoorRHI::float4 sampled =
                classify(sample({origin[0] + direction[0] * distance, origin[1] + direction[1] * distance, origin[2] + direction[2] * distance}));
            const float alpha = 1.0f - std::exp(-sampled.w * volume.stepSize);
            const float weight = transmittance * alpha;
            color = {color[0] + weight * sampled.x, color[1] + weight * sampled.y, color[2] + weight * sampled.z};
            transmittance *= 1.0f - alpha;
        }
        return color;
    }

    const Volume& volume;
};

Camera makeOrbitCamera(const double spiral, const std::uint32_t count, const std::uint32_t imageSize) {
    const double golden = std::numbers::pi * (3.0 - std::sqrt(5.0));
    const double height = 1.0 - 2.0 * spiral / count;
    const double ring = std::sqrt(1.0 - height * height);
    const Vector3 eye{cameraDistance * ring * std::cos(golden * spiral), cameraDistance * height, cameraDistance * ring * std::sin(golden * spiral)};
    const Vector3 forward{-eye[0] / cameraDistance, -eye[1] / cameraDistance, -eye[2] / cameraDistance};
    Vector3 right{-forward[2], 0.0, forward[0]};
    const double rightLength = std::hypot(right[0], right[2]);
    right = {right[0] / rightLength, 0.0, right[2] / rightLength};
    const Vector3 down{forward[1] * right[2] - forward[2] * right[1], forward[2] * right[0] - forward[0] * right[2],
                       forward[0] * right[1] - forward[1] * right[0]};
    const Matrix3 rotation{right, down, forward};
    Vector3 translation{};
    for (std::size_t row = 0; row < 3; ++row)
        translation[row] = -(rotation[row][0] * eye[0] + rotation[row][1] * eye[1] + rotation[row][2] * eye[2]);
    const auto focal = static_cast<float>(0.5 * imageSize / tanHalfFov);
    const float center = 0.5f * static_cast<float>(imageSize);
    return makeCamera(rotation, translation, {focal, focal, center, center}, imageSize, imageSize);
}

std::uint64_t getFingerprint(const Volume& volume, const std::uint32_t imageSize) {
    std::uint64_t hash = 0xcbf29ce484222325ull;
    const auto add = [&](const auto& value) {
        const auto* bytes = reinterpret_cast<const unsigned char*>(&value);
        for (std::size_t index = 0; index < sizeof(value); ++index)
            hash = (hash ^ bytes[index]) * 0x100000001b3ull;
    };
    for (const auto& value : {volume.width, volume.height, volume.depth, imageSize})
        add(value);
    add(volume.stepSize);
    for (const NoorRHI::float4& entry : volume.transferFunction)
        add(entry);
    for (const float value : volume.values)
        add(value);
    return hash;
}

std::expected<void, std::string> renderMissingTargets(const Volume& volume, const std::vector<ViewSource>& views) {
    const ReferenceMarcher marcher(volume);
    std::vector<std::string> errors(views.size());
    parallelFor(views.size(), [&](const std::size_t index) {
        const ViewSource& view = views[index];
        if (std::filesystem::exists(view.file))
            return;
        Core::Image image(view.camera.width, view.camera.height);
        image.setEncoding(Core::Image::Encoding::Srgb);
        marcher.render(view.camera, image);
        const std::filesystem::path partial = std::filesystem::path(view.file).replace_extension(".partial.png");
        if (std::expected<void, std::string> saved = Core::savePng(image, partial); !saved) {
            errors[index] = std::move(saved.error());
            return;
        }
        std::error_code error;
        std::filesystem::rename(partial, view.file, error);
        if (error)
            errors[index] = std::format("{} does not move: {}", partial.string(), error.message());
    });
    for (std::string& error : errors)
        if (!error.empty())
            return std::unexpected(std::move(error));
    return {};
}

std::expected<Dataset, std::string> loadVolumeScene(Volume volume, const std::uint32_t imageSize, const std::filesystem::path& cache,
                                                    const LoadOptions& options) {
    const std::filesystem::path folder = cache / std::format("{:016x}", getFingerprint(volume, imageSize));
    std::error_code error;
    std::filesystem::create_directories(folder, error);
    if (error)
        return std::unexpected(std::format("{} does not open: {}", folder.string(), error.message()));
    DatasetSource source;
    for (std::uint32_t index = 0; index < trainViewCount; ++index)
        source.train.push_back(
            {std::format("train_{:03}", index), makeOrbitCamera(index + 0.5, trainViewCount, imageSize), folder / std::format("train_{:03}.png", index)});
    for (std::uint32_t index = 0; index < testViewCount; ++index)
        source.test.push_back(
            {std::format("test_{:03}", index), makeOrbitCamera(2.0 * index + 1.0, trainViewCount, imageSize), folder / std::format("test_{:03}.png", index)});
    for (const std::vector<ViewSource>* views : {&source.train, &source.test})
        if (std::expected<void, std::string> rendered = renderMissingTargets(volume, *views); !rendered)
            return std::unexpected(std::move(rendered.error()));
    Dataset dataset;
    dataset.points = makeRandomPoints(startPointCount, -0.5f, 0.5f, startPointSeed);
    setBounds(dataset, source.train);
    if (std::expected<void, std::string> loaded = loadViews(dataset, source, options); !loaded)
        return std::unexpected(std::move(loaded.error()));
    dataset.volume = std::move(volume);
    return dataset;
}

float getPercentile(std::vector<float> values, const double fraction) {
    const auto rank = static_cast<std::ptrdiff_t>(fraction * static_cast<double>(values.size() - 1));
    std::ranges::nth_element(values, values.begin() + rank);
    return values[static_cast<std::size_t>(rank)];
}

//! The MSD volumes are skull-stripped: every voxel outside the brain is exactly 0.
std::expected<Volume, std::string> makeBrainVolume(const NiftiVolume& scan) {
    const auto [width, height, depth] = scan.size;
    std::array<std::uint32_t, 3> low{width, height, depth};
    std::array<std::uint32_t, 3> high{};
    std::vector<float> brain;
    for (std::uint32_t z = 0; z < depth; ++z)
        for (std::uint32_t y = 0; y < height; ++y)
            for (std::uint32_t x = 0; x < width; ++x) {
                const float value = scan.values[(std::size_t{z} * height + y) * width + x];
                if (value <= 0.0f)
                    continue;
                brain.push_back(value);
                low = {std::min(low[0], x), std::min(low[1], y), std::min(low[2], z)};
                high = {std::max(high[0], x + 1), std::max(high[1], y + 1), std::max(high[2], z + 1)};
            }
    if (brain.empty())
        return std::unexpected("the brain scan is empty");
    const float minimum = getPercentile(brain, 0.01);
    const float range = getPercentile(brain, 0.995) - minimum;
    const std::array<std::uint32_t, 3> extent{high[0] - low[0], high[1] - low[1], high[2] - low[2]};
    const std::uint32_t size = std::ranges::max(extent);
    Volume volume{.width = size, .height = size, .depth = size, .values = std::vector<float>(std::size_t{size} * size * size)};
    const std::array<std::uint32_t, 3> margin{(size - extent[0]) / 2, (size - extent[2]) / 2, (size - extent[1]) / 2};
    for (std::uint32_t z = 0; z < extent[2]; ++z)
        for (std::uint32_t y = 0; y < extent[1]; ++y)
            for (std::uint32_t x = 0; x < extent[0]; ++x) {
                const float value = scan.values[(std::size_t{z + low[2]} * height + y + low[1]) * width + x + low[0]];
                const std::size_t target = (std::size_t{margin[2] + extent[1] - 1 - y} * size + margin[1] + z) * size + margin[0] + x;
                volume.values[target] = value <= 0.0f ? 0.0f : std::clamp((value - minimum) / range, 0.0f, 1.0f);
            }
    return volume;
}

}

std::expected<Dataset, std::string> loadPhantom(const std::filesystem::path& cache, const LoadOptions& options) {
    const Algorithms::Phantom phantom(phantomSize);
    Volume volume{.width = phantomSize,
                  .height = phantomSize,
                  .depth = phantomSize,
                  .values = phantom.getDensity(),
                  .transferFunction = makePhantomTransferFunction(),
                  .stepSize = 0.5f / phantomSize};
    return loadVolumeScene(std::move(volume), phantomImageSize, cache, options);
}

std::expected<Dataset, std::string> loadBrain(const std::filesystem::path& folder, const std::filesystem::path& cache, const LoadOptions& options) {
    const std::expected<NiftiVolume, std::string> scan = readNifti(folder / "imagesTr/BRATS_001.nii.gz");
    if (!scan)
        return std::unexpected(scan.error());
    std::expected<Volume, std::string> volume = makeBrainVolume(*scan);
    if (!volume)
        return std::unexpected(std::move(volume.error()));
    volume->transferFunction = makeBrainTransferFunction();
    volume->stepSize = 0.5f / static_cast<float>(volume->width);
    return loadVolumeScene(std::move(*volume), brainImageSize, cache, options);
}

}
