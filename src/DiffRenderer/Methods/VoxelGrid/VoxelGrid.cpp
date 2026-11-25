#include "DiffRenderer/Methods/VoxelGrid/VoxelGrid.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <numbers>
#include <utility>

#include "GridEditing.h"
#include "GridEditing.slang.h"
#include "GridTracing.h"
#include "GridTracing.slang.h"

namespace DiffRenderer::Methods {

using namespace NoorRHI;

namespace {

constexpr Description description{
    .id = "grid",
    .name = "Plenoxels",
    .representation = "sparse voxel grid: density and SH degree 2, trilinear",
    .renderer = "emission-absorption ray marching with a fixed step",
    .discontinuities = "none: the field is continuous, edges are steep density",
    .reference = "Fridovich-Keil et al. 2022",
    .scene = true,
    .object = true,
    .volume = true,
};

constexpr std::uint32_t parameterCount = 1 + GridLayout::shCount;
constexpr std::uint32_t maxResolution = 512;
constexpr std::uint32_t maxVoxels = 12'000'000;
constexpr std::uint32_t dilationRadius = 2;
constexpr float initialDensity = 0.1f;
constexpr float variationFraction = 0.01f;
constexpr float objectBound = 1.5f;
constexpr float volumeBound = 0.5f;
//! Plenoxels trains 128000 iterations, but its learning rates decay over 250000.
constexpr double decayFraction = 128000.0 / 250000.0;
constexpr double densityRateStart = 30.0;
constexpr double densityRateEnd = 0.05;
constexpr double densityDelayFraction = 15000.0 / 128000.0;
constexpr double densityDelayFactor = 0.01;
constexpr double shRateStart = 1e-2;
constexpr double shRateEnd = 5e-6;
constexpr Training::AdamSettings rmsProp{.beta1 = 0.0f, .beta2 = 0.95f, .epsilon = 1e-8f};
//! A dispatch has at most 65535 groups in x; the fill loops with a stride of the thread count.
constexpr std::uint32_t maxFillThreads = 65535u * 256u;

std::size_t getCube(const std::uint32_t size) {
    return std::size_t{size} * size * size;
}

std::uint64_t getAddress(const std::uint64_t base, const std::size_t offset) {
    return base + offset * sizeof(float);
}

double getExponentialRate(const double start, const double end, const double progress) {
    return std::exp(std::lerp(std::log(start), std::log(end), progress));
}

}

const Description& VoxelGrid::getDescription() const {
    return description;
}

std::expected<void, Error> VoxelGrid::open(Device& target, const Scenes::Dataset& source, const Settings& chosen) {
    close();
    device = &target;
    dataset = &source;
    settings = chosen;
    contracted = source.kind == Scenes::Kind::Scene;
    switch (source.kind) {
    case Scenes::Kind::Object:
        configuration = {{{{128, 0.0}, {256, 0.1}, {512, 0.3}}}, 3, 1e-5f, 1e-3f, 0.256f};
        placement = {0.0f, 0.0f, 0.0f, 1.0f / objectBound};
        break;
    case Scenes::Kind::Volume:
        configuration = {{{{128, 0.0}, {256, 0.3}}}, 2, 1e-5f, 1e-3f, 0.0256f};
        placement = {0.0f, 0.0f, 0.0f, 1.0f / volumeBound};
        break;
    case Scenes::Kind::Scene:
        configuration = {{{{128, 0.0}, {256, 0.1}, {512, 0.3}}}, 3, 5e-4f, 1e-2f, 0.256f};
        placement = {source.center.x, source.center.y, source.center.z, 1.0f / source.extent};
        break;
    }
    for (std::size_t index = 0; index < configuration.stageCount; ++index)
        stageStarts[index] = static_cast<int>(std::lround(configuration.stages[index].start * chosen.iterations));
    std::size_t pixels = 0;
    for (const auto* views : {&source.train, &source.test})
        for (const Scenes::View& view : *views)
            pixels = std::max(pixels, std::size_t{view.camera.width} * view.camera.height);
    const std::size_t bricks = getCube(maxResolution / GridLayout::brickWidth);
    for (std::expected<void, Error> opened :
         {renderPipeline.open(target, GridTracing::render), backwardPipeline.open(target, GridTracing::backward), fillPipeline.open(target, GridEditing::fill),
          variationPipeline.open(target, GridEditing::totalVariation), keepPipeline.open(target, GridEditing::markKept),
          dilatePipeline.open(target, GridEditing::dilate), countPipeline.open(target, GridEditing::countBricks),
          linkPipeline.open(target, GridEditing::assignLinks), resamplePipeline.open(target, GridEditing::resample),
          occupancyPipeline.open(target, GridEditing::markOccupancy), adam.open(target, rmsProp), scan.open(target, bricks), brickCounts.open(target, bricks),
          brickOffsets.open(target, bricks), lastCount.open(target, 1), lastOffset.open(target, 1), image.open(target, pixels * 3),
          imageGradients.open(target, pixels * 3)})
        if (!opened)
            return opened;
    return resample(configuration.stages[0].resolution);
}

void VoxelGrid::close() {
    *this = VoxelGrid();
}

std::expected<void, Error> VoxelGrid::submit(CommandList&& list) {
    std::expected<Token, Error> token = device->submit(std::move(list));
    if (!token)
        return std::unexpected(std::move(token.error()));
    return device->wait(*token);
}

void VoxelGrid::recordFill(CommandList& list, const std::uint64_t address, const std::size_t count, const uint value) const {
    const auto threads = static_cast<std::uint32_t>(std::clamp<std::size_t>(count, 1, maxFillThreads));
    list.dispatch(fillPipeline, {threads}, GridEditing::FillArguments{Ptr<uint>(address), static_cast<uint>(count), threads, value});
}

void VoxelGrid::recordKeep(CommandList& list, const float threshold, const Ptr<uint> keep, const Ptr<uint> spare) const {
    const std::uint32_t size = grid.resolution;
    list.dispatch(keepPipeline, {size, size, size},
                  GridEditing::KeepArguments{grid.links.getPtr(), Ptr<uint>(maxWeights.getPtr().getAddress()), spare, size, threshold});
    for (const auto [axis, from, to] : {std::tuple{0u, spare, keep}, std::tuple{1u, keep, spare}, std::tuple{2u, spare, keep}})
        list.dispatch(dilatePipeline, {size, size, size}, GridEditing::DilateArguments{from, to, size, axis, dilationRadius});
}

std::expected<std::uint32_t, Error> VoxelGrid::countVoxels(const std::uint32_t resolution, const Ptr<uint> keep) {
    const std::uint32_t bricks = resolution / GridLayout::brickWidth;
    const auto brickCount = static_cast<std::uint32_t>(getCube(bricks));
    CommandList list(*device);
    list.dispatch(countPipeline, {bricks * GridLayout::brickSize, bricks, bricks},
                  GridEditing::LinkArguments{keep, brickCounts.getPtr(), brickOffsets.getPtr(), {}, grid.resolution, resolution});
    if (std::expected<void, Error> scanned = scan.record(list, brickCounts.getPtr(), brickOffsets.getPtr(), brickCount); !scanned)
        return std::unexpected(std::move(scanned.error()));
    list.download(brickCounts, lastCount, brickCount - 1);
    list.download(brickOffsets, lastOffset, brickCount - 1);
    if (std::expected<void, Error> submitted = submit(std::move(list)); !submitted)
        return std::unexpected(std::move(submitted.error()));
    std::expected<std::span<const uint>, Error> count = lastCount.read();
    if (!count)
        return std::unexpected(std::move(count.error()));
    std::expected<std::span<const uint>, Error> offset = lastOffset.read();
    if (!offset)
        return std::unexpected(std::move(offset.error()));
    return count->front() + offset->front();
}

std::expected<void, Error> VoxelGrid::resample(const std::uint32_t resolution) {
    Buffer<uint> keep;
    Buffer<uint> spare;
    if (grid.count > 0)
        for (std::expected<void, Error> opened : {keep.open(*device, getCube(grid.resolution)), spare.open(*device, getCube(grid.resolution))})
            if (!opened)
                return opened;
    float threshold = configuration.weightThreshold / static_cast<float>(grid.resolution);
    std::uint32_t count = 0;
    while (true) {
        if (grid.count > 0) {
            CommandList list(*device);
            recordKeep(list, threshold, keep.getPtr(), spare.getPtr());
            if (std::expected<void, Error> submitted = submit(std::move(list)); !submitted)
                return submitted;
        }
        std::expected<std::uint32_t, Error> counted = countVoxels(resolution, keep.getPtr());
        if (!counted)
            return std::unexpected(std::move(counted.error()));
        count = *counted;
        if (count <= maxVoxels)
            break;
        if (grid.count == 0)
            return fail(ErrorCode::InvalidArgument, std::format("the grid of {}^3 has more than {} voxels", resolution, maxVoxels));
        threshold *= 2.0f;
    }
    if (count == 0)
        return fail(ErrorCode::InvalidState, std::format("the grid of {}^3 keeps no voxel", resolution));
    gradients = {};
    first = {};
    second = {};
    maxWeights = {};
    Grid next;
    next.resolution = resolution;
    next.count = count;
    const std::uint32_t occupancySize = resolution / GridLayout::brickWidth + 1;
    for (std::expected<void, Error> opened : {next.links.open(*device, getCube(resolution)), next.parameters.open(*device, std::size_t{count} * parameterCount),
                                              next.occupancy.open(*device, getCube(occupancySize))})
        if (!opened)
            return opened;
    const auto toLayout = [](const Grid& source) {
        const std::uint64_t parameters = source.parameters ? source.parameters.getPtr().getAddress() : 0;
        return GridLayout::Grid{
            source.links ? source.links.getPtr() : Ptr<int>(),          Ptr<float>(parameters), Ptr<float>(getAddress(parameters, source.count)),
            source.occupancy ? source.occupancy.getPtr() : Ptr<uint>(), source.resolution,      source.resolution / GridLayout::brickWidth + 1};
    };
    const std::uint32_t bricks = resolution / GridLayout::brickWidth;
    CommandList list(*device);
    list.dispatch(linkPipeline, {bricks * GridLayout::brickSize, bricks, bricks},
                  GridEditing::LinkArguments{keep.getPtr(), brickCounts.getPtr(), brickOffsets.getPtr(), next.links.getPtr(), grid.resolution, resolution});
    list.dispatch(resamplePipeline, {resolution, resolution, resolution}, GridEditing::ResampleArguments{toLayout(grid), toLayout(next), initialDensity});
    recordFill(list, next.occupancy.getPtr().getAddress(), next.occupancy.getSize(), 0);
    list.dispatch(occupancyPipeline, {resolution, resolution, resolution}, GridEditing::OccupancyArguments{toLayout(next)});
    if (std::expected<void, Error> submitted = submit(std::move(list)); !submitted)
        return submitted;
    grid = std::move(next);
    const std::size_t values = std::size_t{count} * parameterCount;
    for (std::expected<void, Error> opened :
         {gradients.open(*device, values), first.open(*device, values), second.open(*device, values), maxWeights.open(*device, count)})
        if (!opened)
            return opened;
    CommandList clear(*device);
    recordFill(clear, first.getPtr().getAddress(), values, 0);
    recordFill(clear, second.getPtr().getAddress(), values, 0);
    recordFill(clear, maxWeights.getPtr().getAddress(), count, 0);
    optimizerStep = 0;
    return submit(std::move(clear));
}

void VoxelGrid::recordTrace(CommandList& list, const ComputePipeline& pipeline, const Scenes::Camera& camera, const Trace trace) const {
    const bool training = trace != Trace::Evaluate;
    const float3 origin = camera.getPosition();
    const std::uint64_t parameters = grid.parameters.getPtr().getAddress();
    const std::uint64_t gradientAddress = training ? gradients.getPtr().getAddress() : 0;
    const GridTracing::Arguments arguments{
        .rows = {camera.worldToCamera[0], camera.worldToCamera[1], camera.worldToCamera[2]},
        .intrinsics = camera.intrinsics,
        .origin = {origin.x, origin.y, origin.z, 0.0f},
        .placement = placement,
        .background = {dataset->background.x, dataset->background.y, dataset->background.z, 0.0f},
        .grid = {grid.links.getPtr(), Ptr<float>(parameters), Ptr<float>(getAddress(parameters, grid.count)), grid.occupancy.getPtr(), grid.resolution,
                 grid.resolution / GridLayout::brickWidth + 1},
        .image = image.getPtr(),
        .imageGradients = imageGradients.getPtr(),
        .densityGradients = AtomicPtr<float>(gradientAddress),
        .shGradients = AtomicPtr<float>(training ? getAddress(gradientAddress, grid.count) : 0),
        .maxWeights = trace == Trace::TrainAndWeigh ? maxWeights.getPtr() : AtomicPtr<uint>(),
        .width = camera.width,
        .height = camera.height,
        .contracted = contracted ? 1u : 0u,
        .maxSamples = 4 * grid.resolution,
        .stepSize = 1.0f / static_cast<float>(grid.resolution),
    };
    list.dispatch(pipeline, {camera.width, camera.height}, arguments);
}

double VoxelGrid::getProgress() const {
    return std::clamp(static_cast<double>(iteration) / std::max(settings.iterations, 1), 0.0, 1.0);
}

void VoxelGrid::recordOptimizer(CommandList& list) {
    const double progress = getProgress();
    const double delay = densityDelayFactor + (1.0 - densityDelayFactor) * std::sin(0.5 * std::numbers::pi * std::min(progress / densityDelayFraction, 1.0));
    const double scale = settings.learningRateScale;
    const std::uint64_t parameters = grid.parameters.getPtr().getAddress();
    const std::uint64_t gradientAddress = gradients.getPtr().getAddress();
    const std::uint64_t firstAddress = first.getPtr().getAddress();
    const std::uint64_t secondAddress = second.getPtr().getAddress();
    const auto group = [&](const std::size_t offset, const std::size_t count, const double rate) {
        return Training::AdamGroup{Ptr<float>(getAddress(parameters, offset)),
                                   Ptr<float>(getAddress(gradientAddress, offset)),
                                   Ptr<float>(getAddress(firstAddress, offset)),
                                   Ptr<float>(getAddress(secondAddress, offset)),
                                   static_cast<std::uint32_t>(count),
                                   static_cast<float>(rate * scale)};
    };
    ++optimizerStep;
    adam.record(list, group(0, grid.count, delay * getExponentialRate(densityRateStart, densityRateEnd, progress * decayFraction)), optimizerStep);
    adam.record(list, group(grid.count, std::size_t{grid.count} * GridLayout::shCount, getExponentialRate(shRateStart, shRateEnd, progress * decayFraction)),
                optimizerStep);
}

std::expected<double, Error> VoxelGrid::step(const Target& target, Training::Loss& loss) {
    if (stage + 1 < configuration.stageCount && iteration >= stageStarts[stage + 1]) {
        ++stage;
        if (std::expected<void, Error> resampled = resample(configuration.stages[stage].resolution); !resampled)
            return std::unexpected(std::move(resampled.error()));
    }
    const Scenes::Camera& camera = target.view->camera;
    const std::size_t values = std::size_t{grid.count} * parameterCount;
    const float voxels = static_cast<float>(grid.count) * variationFraction;
    const auto weightWindow = static_cast<int>(dataset->train.size());
    CommandList list(*device);
    recordFill(list, gradients.getPtr().getAddress(), values, 0);
    const bool weighing = stage + 1 < configuration.stageCount && iteration >= stageStarts[stage + 1] - weightWindow;
    if (weighing && iteration == stageStarts[stage + 1] - weightWindow)
        recordFill(list, maxWeights.getPtr().getAddress(), grid.count, 0);
    const Trace trace = weighing ? Trace::TrainAndWeigh : Trace::Train;
    recordTrace(list, renderPipeline, camera, trace);
    loss.record(list, image.getPtr(), target.image, imageGradients.getPtr(), camera.width, camera.height);
    recordTrace(list, backwardPipeline, camera, trace);
    const std::uint64_t gradientAddress = gradients.getPtr().getAddress();
    const std::uint64_t parameters = grid.parameters.getPtr().getAddress();
    list.dispatch(variationPipeline, {grid.resolution, grid.resolution, grid.resolution},
                  GridEditing::VariationArguments{{grid.links.getPtr(), Ptr<float>(parameters), Ptr<float>(getAddress(parameters, grid.count)),
                                                   grid.occupancy.getPtr(), grid.resolution, grid.resolution / GridLayout::brickWidth + 1},
                                                  AtomicPtr<float>(gradientAddress),
                                                  AtomicPtr<float>(getAddress(gradientAddress, grid.count)),
                                                  configuration.densityVariation / voxels,
                                                  configuration.shVariation / voxels,
                                                  variationFraction,
                                                  static_cast<uint>(iteration) * 0x9e3779b9u});
    recordOptimizer(list);
    loss.recordReadback(list);
    if (std::expected<void, Error> submitted = submit(std::move(list)); !submitted)
        return std::unexpected(std::move(submitted.error()));
    ++iteration;
    return loss.read();
}

std::expected<void, Error> VoxelGrid::render(const Scenes::Camera& camera, Core::Image& target) {
    const std::size_t values = std::size_t{camera.width} * camera.height * 3;
    if (values > image.getSize())
        return fail(ErrorCode::InvalidArgument, "the camera is larger than the views of the dataset");
    if (imageReadback.getSize() != values)
        if (std::expected<void, Error> opened = imageReadback.open(*device, values); !opened)
            return opened;
    CommandList list(*device);
    recordTrace(list, renderPipeline, camera, Trace::Evaluate);
    list.download(image, imageReadback);
    if (std::expected<void, Error> submitted = submit(std::move(list)); !submitted)
        return submitted;
    std::expected<std::span<const float>, Error> pixels = imageReadback.read();
    if (!pixels)
        return std::unexpected(std::move(pixels.error()));
    std::ranges::transform(*pixels, target.getPixels().begin(), [](const float value) { return std::clamp(value, 0.0f, 1.0f); });
    return {};
}

Statistics VoxelGrid::getStatistics() const {
    const std::size_t bytes = grid.links.getByteSize() + grid.parameters.getByteSize() + grid.occupancy.getByteSize() + gradients.getByteSize()
                              + first.getByteSize() + second.getByteSize() + maxWeights.getByteSize();
    return {grid.count, bytes};
}

}
