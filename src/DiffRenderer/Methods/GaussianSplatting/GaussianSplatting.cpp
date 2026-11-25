#include "DiffRenderer/Methods/GaussianSplatting/GaussianSplatting.h"

#include <algorithm>
#include <bit>
#include <utility>

#include "TileSplatting.slang.h"

namespace DiffRenderer::Methods {

using namespace NoorRHI;

namespace {

const Description description{
    .id = "3dgs",
    .name = "3D Gaussian Splatting",
    .representation = "3D Gaussians, SH degree 3",
    .renderer = "tile-based splatting",
    .discontinuities = "none: every Gaussian is smooth; occlusion acts through the alpha blending",
    .reference = "Kerbl et al. 2023, diff-gaussian-rasterization",
    .scene = true,
    .object = true,
    .volume = true,
};

constexpr std::uint32_t tileSize = 16;
constexpr std::uint32_t maxInstances = 65535u * 4096u;
constexpr double instanceGrowth = 1.5;
constexpr double instanceFill = 0.8;

std::uint32_t getTileCount(const std::uint32_t size) {
    return (size + tileSize - 1) / tileSize;
}

std::uint32_t grown(const std::uint64_t needed, const double factor, const std::uint64_t limit) {
    return static_cast<std::uint32_t>(std::min<std::uint64_t>(static_cast<std::uint64_t>(static_cast<double>(needed) * factor) + 1, limit));
}

TileSplatting::View getView(const Scenes::Camera& camera, const float4 background) {
    const float3 position = camera.getPosition();
    return {{camera.worldToCamera[0], camera.worldToCamera[1], camera.worldToCamera[2]},
            camera.intrinsics,
            {position.x, position.y, position.z, 0.0f},
            background,
            camera.width,
            camera.height,
            getTileCount(camera.width),
            getTileCount(camera.height)};
}

}

const Description& GaussianSplatting::getDescription() const {
    return description;
}

std::expected<void, Error> GaussianSplatting::open(Device& target, const Scenes::Dataset& dataset, const Settings& chosen) {
    close();
    device = &target;
    if (std::expected<void, Error> opened = model.open(target, dataset, chosen); !opened)
        return opened;
    background = {dataset.background.x, dataset.background.y, dataset.background.z, 0.0f};
    for (const auto* views : {&dataset.train, &dataset.test})
        for (const Scenes::View& view : *views) {
            maxPixels = std::max(maxPixels, view.camera.width * view.camera.height);
            maxTiles = std::max(maxTiles, getTileCount(view.camera.width) * getTileCount(view.camera.height));
        }
    if (std::expected<void, Error> opened = openPipelines(); !opened)
        return opened;
    if (std::expected<void, Error> opened = openWorkBuffers(model.getCapacity()); !opened)
        return opened;
    for (std::expected<void, Error> step :
         {instanceCounts.open(target, 2), instanceTotalReadback.open(target, 1), ranges.open(target, std::size_t{maxTiles} * 2),
          image.open(target, std::size_t{maxPixels} * 3), finalTransmittance.open(target, maxPixels), lastContributors.open(target, maxPixels),
          imageGradient.open(target, std::size_t{maxPixels} * 3), imageReadback.open(target, std::size_t{maxPixels} * 3)})
        if (!step)
            return step;
    return openInstanceBuffers(grown(model.getCount(), 64.0, maxInstances));
}

std::expected<void, Error> GaussianSplatting::openPipelines() {
    for (auto [pipeline, shader] : std::array<std::pair<ComputePipeline*, const Shader*>, 8>{{{&preprocess, &TileSplatting::preprocess},
                                                                                              {&countInstances, &TileSplatting::countInstances},
                                                                                              {&duplicateKeys, &TileSplatting::duplicateKeys},
                                                                                              {&clearRanges, &TileSplatting::clearRanges},
                                                                                              {&identifyRanges, &TileSplatting::identifyRanges},
                                                                                              {&blend, &TileSplatting::blend},
                                                                                              {&blendBackward, &TileSplatting::blendBackward},
                                                                                              {&preprocessBackward, &TileSplatting::preprocessBackward}}})
        if (std::expected<void, Error> opened = pipeline->open(*device, *shader); !opened)
            return opened;
    return {};
}

std::expected<void, Error> GaussianSplatting::openWorkBuffers(const std::uint32_t chosen) {
    workCapacity = chosen;
    const std::size_t size = workCapacity;
    for (std::expected<void, Error> step : {projected.open(*device, size), conics.open(*device, size), colors.open(*device, size * 3),
                                            tilesTouched.open(*device, size), offsets.open(*device, size), splatGradients.open(*device, size * 9),
                                            scan.open(*device, size)})
        if (!step)
            return step;
    return {};
}

std::expected<void, Error> GaussianSplatting::openInstanceBuffers(const std::uint32_t chosen) {
    instanceCapacity = chosen;
    if (std::expected<void, Error> opened = keys.open(*device, instanceCapacity); !opened)
        return opened;
    if (std::expected<void, Error> opened = values.open(*device, instanceCapacity); !opened)
        return opened;
    return sort.open(*device, instanceCapacity);
}

std::expected<void, Error> GaussianSplatting::submit(CommandList&& list) const {
    std::expected<Token, Error> token = device->submit(std::move(list));
    if (!token)
        return std::unexpected(std::move(token.error()));
    return device->wait(*token);
}

TileSplatting::Parameters GaussianSplatting::getPointers(Buffer<float> GaussianModel::GroupBuffers::* member) const {
    const auto get = [&](const GaussianModel::Group group) { return (model.getGroup(group).*member).getPtr(); };
    return {get(GaussianModel::Means),     get(GaussianModel::FeaturesDc), get(GaussianModel::FeaturesRest),
            get(GaussianModel::Opacities), get(GaussianModel::Scales),     get(GaussianModel::Rotations)};
}

std::expected<void, Error> GaussianSplatting::recordForward(CommandList& list, const Scenes::Camera& camera, const bool accumulate) {
    if (camera.width * camera.height > maxPixels || getTileCount(camera.width) * getTileCount(camera.height) > maxTiles)
        return fail(ErrorCode::InvalidArgument, "the camera is larger than the views that 3DGS opened with");
    const std::uint32_t count = model.getCount();
    arguments = {getView(camera, background),
                 getPointers(&GaussianModel::GroupBuffers::parameters),
                 getPointers(&GaussianModel::GroupBuffers::gradients),
                 projected.getPtr(),
                 conics.getPtr(),
                 colors.getPtr(),
                 tilesTouched.getPtr(),
                 offsets.getPtr(),
                 keys.getPtr().as<uint>(),
                 values.getPtr(),
                 instanceCounts.getPtr(),
                 instanceCounts.getPtr() + 1,
                 ranges.getPtr(),
                 image.getPtr(),
                 finalTransmittance.getPtr(),
                 lastContributors.getPtr(),
                 imageGradient.getPtr(),
                 splatGradients.getPtr(),
                 model.getGradientSums(),
                 model.getGradientCounts(),
                 count,
                 GaussianModel::getThreads(count),
                 instanceCapacity,
                 model.getShDegree(),
                 accumulate ? 1u : 0u};
    const std::uint32_t tiles = arguments.view.tilesX * arguments.view.tilesY;
    list.dispatch(preprocess, {arguments.threads}, arguments);
    if (std::expected<void, Error> scanned = scan.record(list, tilesTouched.getPtr(), offsets.getPtr(), count); !scanned)
        return scanned;
    list.dispatch(countInstances, {1}, arguments);
    list.dispatch(duplicateKeys, {arguments.threads}, arguments);
    if (std::expected<void, Error> sorted =
            sort.record(list, keys.getPtr(), values.getPtr(), instanceCapacity, 32 + std::bit_width(tiles), instanceCounts.getPtr());
        !sorted)
        return sorted;
    list.dispatch(clearRanges, {tiles}, arguments);
    TileSplatting::Arguments perInstance = arguments;
    perInstance.threads = GaussianModel::getThreads(instanceCapacity);
    list.dispatch(identifyRanges, {perInstance.threads}, perInstance);
    list.dispatch(blend, {arguments.view.tilesX * tileSize, arguments.view.tilesY * tileSize}, arguments);
    list.download(instanceCounts, instanceTotalReadback, 1);
    return {};
}

void GaussianSplatting::recordBackward(CommandList& list) const {
    list.dispatch(blendBackward, {arguments.view.tilesX * tileSize, arguments.view.tilesY * tileSize}, arguments);
    list.dispatch(preprocessBackward, {arguments.threads}, arguments);
    model.recordUpdate(list);
}

std::expected<void, Error> GaussianSplatting::readInstanceTotal() {
    std::expected<std::span<const uint>, Error> total = instanceTotalReadback.read();
    if (!total)
        return std::unexpected(std::move(total.error()));
    instanceTotal = total->front();
    if (instanceTotal > maxInstances)
        return fail(ErrorCode::InvalidState, "3DGS needs more tile instances than the radix sort supports");
    if (static_cast<double>(instanceTotal) <= instanceFill * instanceCapacity)
        return {};
    return openInstanceBuffers(grown(instanceTotal, instanceGrowth, maxInstances));
}

std::expected<double, Error> GaussianSplatting::step(const Target& target, Training::Loss& loss) {
    const GaussianModel::Phase phase = model.beginStep();
    const Scenes::Camera& camera = target.view->camera;
    CommandList list(*device);
    if (std::expected<void, Error> recorded = recordForward(list, camera, phase.densifying); !recorded)
        return std::unexpected(std::move(recorded.error()));
    loss.record(list, image.getPtr(), target.image, imageGradient.getPtr(), camera.width, camera.height);
    recordBackward(list);
    loss.recordReadback(list);
    if (std::expected<void, Error> submitted = submit(std::move(list)); !submitted)
        return std::unexpected(std::move(submitted.error()));
    std::expected<double, Error> value = loss.read();
    if (!value)
        return value;
    if (std::expected<void, Error> read = readInstanceTotal(); !read)
        return std::unexpected(std::move(read.error()));
    if (std::expected<void, Error> finished = model.finishStep(phase); !finished)
        return std::unexpected(std::move(finished.error()));
    if (model.getCapacity() > workCapacity)
        if (std::expected<void, Error> opened = openWorkBuffers(model.getCapacity()); !opened)
            return std::unexpected(std::move(opened.error()));
    return value;
}

std::expected<void, Error> GaussianSplatting::render(const Scenes::Camera& camera, Core::Image& result) {
    for (;;) {
        const std::uint32_t usedCapacity = instanceCapacity;
        CommandList list(*device);
        if (std::expected<void, Error> recorded = recordForward(list, camera, false); !recorded)
            return recorded;
        list.download(image, imageReadback);
        if (std::expected<void, Error> submitted = submit(std::move(list)); !submitted)
            return submitted;
        if (std::expected<void, Error> read = readInstanceTotal(); !read)
            return read;
        if (instanceTotal <= usedCapacity)
            break;
    }
    std::expected<std::span<const float>, Error> pixels = imageReadback.read();
    if (!pixels)
        return std::unexpected(std::move(pixels.error()));
    const std::span<float> destination = result.getPixels();
    for (std::size_t index = 0; index < destination.size(); ++index)
        destination[index] = std::clamp((*pixels)[index], 0.0f, 1.0f);
    return {};
}

std::expected<std::optional<TrainedGaussians>, Error> GaussianSplatting::readGaussians() {
    return model.readGaussians();
}

}
