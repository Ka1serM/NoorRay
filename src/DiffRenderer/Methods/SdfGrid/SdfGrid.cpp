#include "DiffRenderer/Methods/SdfGrid/SdfGrid.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <format>
#include <utility>

#include "DiffRenderer/Methods/SdfGrid/IsoSurface.h"
#include "SurfaceRendering.slang.h"

namespace DiffRenderer::Methods {

using namespace NoorRHI;

namespace {

//! Wang et al. 2024 start at 32^3 and double the grid at 10, 20, 40 and 60 % of the iterations; each stage has about edge^3 nodes.
constexpr std::array<std::uint32_t, 4> stageEdges{32, 64, 128, 224};
constexpr std::array<float, 4> stageStarts{0.0f, 0.1f, 0.2f, 0.4f};
//! The SH coefficients become 4 and then 9 at these fractions of the iterations.
constexpr std::array<float, 2> shStarts{0.3f, 0.5f};
constexpr std::uint32_t environmentWidth = 512;
constexpr std::uint32_t environmentHeight = 256;
constexpr std::uint32_t maxThreads = 65535u * 256u;
constexpr std::uint32_t renderSamplesPerSide = 4;
constexpr float bandPixels = 1.0f;
constexpr float minimumStep = 0.05f;
constexpr std::uint32_t trainSamplesPerSide = 2;
//! Wang et al. 2024 use 0.01 in the unit cube at 32^3 and halve it with each doubling of the grid: 0.32 voxels.
constexpr float sdfLearningRate = 0.32f;
//! The radius of the start sphere relative to the shortest side of the bounds; 0.3 in the unit cube in Wang et al. 2024.
constexpr float startRadius = 0.3f;
constexpr float colourLearningRate = 0.05f;
constexpr float environmentLearningRate = 0.05f;
constexpr float finalLearningRateFactor = 0.1f;
constexpr float curvatureWeight = 0.05f;
constexpr float emptyWeight = 10.0f;
constexpr float objectPadding = 1.15f;
constexpr float scenePercentile = 0.05f;
constexpr float cropBand = 2.0f;
constexpr float cropPadding = 4.0f;

const Description description{
    .id = "sdf",
    .name = "SDF grid (relaxed boundary)",
    .representation = "SDF + SH colour voxel grid",
    .renderer = "sphere tracing of the opaque zero level set",
    .discontinuities = "relaxed boundary: rays that pass within one pixel footprint of the surface add the silhouette term",
    .reference = "Wang et al. 2024 (A Simple Approach to Differentiable Rendering of SDFs)",
    .scene = true,
    .object = true,
    .hasBoundaryTerm = true,
};

std::uint32_t getThreads(const std::size_t count) {
    return static_cast<std::uint32_t>(std::clamp<std::size_t>(count, 1, maxThreads));
}

float3 operator+(const float3 a, const float3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

float3 operator-(const float3 a, const float3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

float3 operator*(const float3 a, const float b) {
    return {a.x * b, a.y * b, a.z * b};
}

SurfaceRendering::Grid getGrid(const float3 lower, const float3 upper, const std::uint32_t edge) {
    const float3 extent = upper - lower;
    const float voxel = std::cbrt(extent.x * extent.y * extent.z / static_cast<float>(edge * edge * edge));
    const auto getCount = [voxel](const float length) { return std::max(2u, static_cast<std::uint32_t>(std::ceil(length / voxel)) + 1); };
    const uint4 size{getCount(extent.x), getCount(extent.y), getCount(extent.z), 0};
    const float3 center = (lower + upper) * 0.5f;
    const float3 half{static_cast<float>(size.x - 1) * voxel * 0.5f, static_cast<float>(size.y - 1) * voxel * 0.5f,
                      static_cast<float>(size.z - 1) * voxel * 0.5f};
    const float3 origin = center - half;
    return {{origin.x, origin.y, origin.z, voxel}, {size.x, size.y, size.z, size.x * size.y * size.z}};
}

float3 getNode(const SurfaceRendering::Grid& grid, const std::uint32_t x, const std::uint32_t y, const std::uint32_t z) {
    return {grid.origin.x + static_cast<float>(x) * grid.origin.w, grid.origin.y + static_cast<float>(y) * grid.origin.w,
            grid.origin.z + static_cast<float>(z) * grid.origin.w};
}

float getComponent(const float3 value, const int axis) {
    return axis == 0 ? value.x : (axis == 1 ? value.y : value.z);
}

float getPercentile(std::vector<float>& values, const float fraction) {
    const auto position = values.begin() + static_cast<std::ptrdiff_t>(fraction * static_cast<float>(values.size() - 1));
    std::ranges::nth_element(values, position);
    return *position;
}

}

std::expected<void, Error> GridParameters::open(Device& device, const std::size_t count) {
    for (std::expected<void, Error> opened : {values.open(device, count), gradients.open(device, count), first.open(device, count), second.open(device, count)})
        if (!opened)
            return opened;
    return {};
}

Training::AdamGroup GridParameters::getGroup(const float learningRate) const {
    return {values.getPtr(), getGradientPtr(), first.getPtr(), second.getPtr(), static_cast<std::uint32_t>(values.getSize()), learningRate};
}

const Description& SdfGrid::getDescription() const {
    return description;
}

std::expected<void, Error> SdfGrid::open(Device& target, const Scenes::Dataset& dataset, const Settings& chosen) {
    if (dataset.kind == Scenes::Kind::Volume)
        return fail(ErrorCode::InvalidArgument, "the SDF grid does not support volume scenarios");
    if (dataset.points.positions.empty())
        return fail(ErrorCode::InvalidArgument, "the SDF grid needs the start points for its bounds");
    device = &target;
    settings = chosen;
    stage = 0;
    iteration = 0;
    adamStep = 0;
    closedSdf.clear();
    for (auto [pipeline, shader] : std::array<std::pair<ComputePipeline*, const Shader*>, 8>{{{&renderPipeline, &SurfaceRendering::render},
                                                                                              {&backwardPipeline, &SurfaceRendering::backward},
                                                                                              {&seedPipeline, &SurfaceRendering::seed},
                                                                                              {&floodPipeline, &SurfaceRendering::flood},
                                                                                              {&redistancePipeline, &SurfaceRendering::redistance},
                                                                                              {&smoothPipeline, &SurfaceRendering::smooth},
                                                                                              {&clearPipeline, &SurfaceRendering::clear},
                                                                                              {&resamplePipeline, &SurfaceRendering::resample}}})
        if (std::expected<void, Error> opened = pipeline->open(target, *shader); !opened)
            return opened;
    if (std::expected<void, Error> opened = adam.open(target); !opened)
        return opened;
    cropping = dataset.kind == Scenes::Kind::Object;
    for (int axis = 0; axis < 3; ++axis) {
        std::vector<float> values;
        values.reserve(dataset.points.positions.size());
        for (const float3 position : dataset.points.positions)
            values.push_back(getComponent(position, axis) - getComponent(dataset.center, axis));
        const float fraction = cropping ? 0.0f : scenePercentile;
        float lower = getPercentile(values, fraction);
        float upper = getPercentile(values, 1.0f - fraction);
        if (cropping) {
            lower *= objectPadding;
            upper *= objectPadding;
        }
        (axis == 0 ? boundsLower.x : (axis == 1 ? boundsLower.y : boundsLower.z)) = lower + getComponent(dataset.center, axis);
        (axis == 0 ? boundsUpper.x : (axis == 1 ? boundsUpper.y : boundsUpper.z)) = upper + getComponent(dataset.center, axis);
    }
    background = cropping ? float4{dataset.background.x, dataset.background.y, dataset.background.z, 0.0f} : float4{0.0f, 0.0f, 0.0f, 1.0f};
    const float3 extent = boundsUpper - boundsLower;
    grid = getGrid(boundsLower, boundsUpper, stageEdges.front());
    if (std::expected<void, Error> opened = openGrid(grid, nullptr, nullptr, grid); !opened)
        return opened;
    const float3 center = (boundsLower + boundsUpper) * 0.5f;
    const float radius = startRadius * std::min({extent.x, extent.y, extent.z});
    std::vector<float> sphere(grid.size.w);
    for (std::uint32_t z = 0; z < grid.size.z; ++z)
        for (std::uint32_t y = 0; y < grid.size.y; ++y)
            for (std::uint32_t x = 0; x < grid.size.x; ++x) {
                const float3 offset = getNode(grid, x, y, z) - center;
                sphere[(std::size_t{z} * grid.size.y + y) * grid.size.x + x] =
                    std::sqrt(offset.x * offset.x + offset.y * offset.y + offset.z * offset.z) - radius;
            }
    if (std::expected<void, Error> uploaded = sdf.values.upload(sphere); !uploaded)
        return uploaded;
    const std::size_t environmentCount = cropping ? 1 : std::size_t{environmentWidth} * environmentHeight * 3;
    if (std::expected<void, Error> opened = environment.open(target, environmentCount); !opened)
        return opened;
    CommandList list(target);
    for (const Ptr<float> pointer : {environment.values.getPtr(), environment.first.getPtr(), environment.second.getPtr()})
        recordClear(list, pointer, environment.values.getSize());
    if (std::expected<void, Error> submitted = submit(std::move(list)); !submitted)
        return submitted;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    for (const Scenes::View& view : dataset.train) {
        width = std::max(width, view.camera.width);
        height = std::max(height, view.camera.height);
    }
    return openImage(width, height);
}

std::expected<void, Error> SdfGrid::openGrid(const SurfaceRendering::Grid& chosen, const GridParameters* sourceSdf, const GridParameters* sourceColour,
                                             const SurfaceRendering::Grid& sourceGrid) {
    GridParameters newSdf;
    GridParameters newColour;
    if (std::expected<void, Error> opened = newSdf.open(*device, chosen.size.w); !opened)
        return opened;
    if (std::expected<void, Error> opened = newColour.open(*device, std::size_t{chosen.size.w} * SurfaceRendering::colourChannels); !opened)
        return opened;
    for (Buffer<float4>& buffer : seeds)
        if (std::expected<void, Error> opened = buffer.open(*device, chosen.size.w); !opened)
            return opened;
    CommandList list(*device);
    for (const GridParameters* parameters : {&newSdf, &newColour}) {
        recordClear(list, parameters->first.getPtr(), parameters->first.getSize());
        recordClear(list, parameters->second.getPtr(), parameters->second.getSize());
    }
    if (sourceSdf && sourceColour) {
        SurfaceRendering::Arguments arguments{};
        arguments.grid = chosen;
        arguments.source = sourceGrid;
        arguments.sdf = newSdf.values.getPtr();
        arguments.colour = newColour.values.getPtr();
        arguments.sourceSdf = sourceSdf->values.getPtr();
        arguments.sourceColour = sourceColour->values.getPtr();
        arguments.threads = getThreads(chosen.size.w);
        list.dispatch(resamplePipeline, {arguments.threads}, arguments);
    } else {
        recordClear(list, newColour.values.getPtr(), newColour.values.getSize());
    }
    if (std::expected<void, Error> submitted = submit(std::move(list)); !submitted)
        return submitted;
    sdf = std::move(newSdf);
    colour = std::move(newColour);
    grid = chosen;
    adamStep = 0;
    return {};
}

std::expected<void, Error> SdfGrid::refine(const std::uint32_t edge) {
    float3 lower = boundsLower;
    float3 upper = boundsUpper;
    if (cropping) {
        std::expected<std::vector<float>, Error> values = downloadSdf();
        if (!values)
            return std::unexpected(std::move(values.error()));
        std::array<std::uint32_t, 3> first{grid.size.x, grid.size.y, grid.size.z};
        std::array<std::uint32_t, 3> last{0, 0, 0};
        const float band = cropBand * grid.origin.w;
        for (std::uint32_t z = 0; z < grid.size.z; ++z)
            for (std::uint32_t y = 0; y < grid.size.y; ++y)
                for (std::uint32_t x = 0; x < grid.size.x; ++x) {
                    if ((*values)[(std::size_t{z} * grid.size.y + y) * grid.size.x + x] > band)
                        continue;
                    first = {std::min(first[0], x), std::min(first[1], y), std::min(first[2], z)};
                    last = {std::max(last[0], x), std::max(last[1], y), std::max(last[2], z)};
                }
        if (first[0] > last[0])
            return fail(ErrorCode::InvalidState, "the SDF grid lost its surface before the refinement");
        const float padding = cropPadding * grid.origin.w;
        const float3 low = getNode(grid, first[0], first[1], first[2]) - float3{padding, padding, padding};
        const float3 high = getNode(grid, last[0], last[1], last[2]) + float3{padding, padding, padding};
        lower = {std::max(low.x, boundsLower.x), std::max(low.y, boundsLower.y), std::max(low.z, boundsLower.z)};
        upper = {std::min(high.x, boundsUpper.x), std::min(high.y, boundsUpper.y), std::min(high.z, boundsUpper.z)};
    }
    const GridParameters oldSdf = std::move(sdf);
    const GridParameters oldColour = std::move(colour);
    return openGrid(getGrid(lower, upper, edge), &oldSdf, &oldColour, grid);
}

std::expected<void, Error> SdfGrid::openImage(const std::uint32_t width, const std::uint32_t height) {
    const std::size_t pixels = std::size_t{width} * height;
    for (std::expected<void, Error> opened :
         {image.open(*device, pixels * 3), imageGradient.open(*device, pixels * 3), imageReadback.open(*device, pixels * 3)})
        if (!opened)
            return opened;
    imageWidth = width;
    imageHeight = height;
    return {};
}

std::expected<std::vector<float>, Error> SdfGrid::downloadSdf() const {
    std::vector<float> values(grid.size.w);
    if (std::expected<void, Error> downloaded = sdf.values.download(values); !downloaded)
        return std::unexpected(std::move(downloaded.error()));
    return values;
}

std::expected<void, Error> SdfGrid::submit(CommandList&& list) const {
    std::expected<Token, Error> token = device->submit(std::move(list));
    if (!token)
        return std::unexpected(std::move(token.error()));
    return device->wait(*token);
}

void SdfGrid::recordClear(CommandList& list, const Ptr<float> target, const std::size_t count) const {
    SurfaceRendering::Arguments arguments{};
    arguments.clearTarget = target;
    arguments.count = static_cast<std::uint32_t>(count);
    arguments.threads = getThreads(count);
    list.dispatch(clearPipeline, {arguments.threads}, arguments);
}

//! Wang et al. 2024 redistance the SDF after every step with fast sweeping; jump flooding with one extra unit pass gives the same distances on the GPU.
void SdfGrid::recordRedistance(CommandList& list) {
    SurfaceRendering::Arguments arguments{};
    arguments.grid = grid;
    arguments.sdf = sdf.values.getPtr();
    arguments.threads = getThreads(grid.size.w);
    arguments.nextSeeds = seeds[0].getPtr();
    list.dispatch(seedPipeline, {arguments.threads}, arguments);
    std::size_t current = 0;
    const auto flood = [&](const std::uint32_t jump) {
        arguments.jump = jump;
        arguments.seeds = seeds[current].getPtr();
        arguments.nextSeeds = seeds[1 - current].getPtr();
        list.dispatch(floodPipeline, {arguments.threads}, arguments);
        current = 1 - current;
    };
    for (std::uint32_t jump = std::bit_floor(std::max({grid.size.x, grid.size.y, grid.size.z}) - 1); jump > 0; jump /= 2)
        flood(jump);
    flood(1);
    arguments.seeds = seeds[current].getPtr();
    list.dispatch(redistancePipeline, {arguments.threads}, arguments);
}

float SdfGrid::getProgress() const {
    return std::min(1.0f, static_cast<float>(iteration) / static_cast<float>(std::max(1, settings.iterations)));
}

SurfaceRendering::Arguments SdfGrid::getArguments(const Scenes::Camera& camera, const std::uint32_t samplesPerSide) const {
    const auto& [right, down, forward] = camera.worldToCamera;
    const float3 position = camera.getPosition();
    const float progress = getProgress();
    SurfaceRendering::Arguments arguments{};
    arguments.cameraRight = {right.x, right.y, right.z, 0.0f};
    arguments.cameraDown = {down.x, down.y, down.z, 0.0f};
    arguments.cameraForward = {forward.x, forward.y, forward.z, 0.0f};
    arguments.cameraPosition = {position.x, position.y, position.z, 0.0f};
    arguments.intrinsics = camera.intrinsics;
    arguments.background = background;
    arguments.grid = grid;
    arguments.sdf = sdf.values.getPtr();
    arguments.colour = colour.values.getPtr();
    arguments.environment = environment.values.getPtr();
    arguments.sdfGradient = sdf.gradients.getPtr();
    arguments.colourGradient = colour.gradients.getPtr();
    arguments.environmentGradient = environment.gradients.getPtr();
    arguments.image = image.getPtr();
    arguments.imageGradient = imageGradient.getPtr();
    arguments.width = camera.width;
    arguments.height = camera.height;
    arguments.environmentWidth = environmentWidth;
    arguments.environmentHeight = environmentHeight;
    arguments.step = minimumStep * grid.origin.w;
    arguments.curvatureWeight = curvatureWeight;
    arguments.emptyWeight = emptyWeight;
    arguments.samplesPerSide = samplesPerSide;
    arguments.boundaryTerm = settings.boundaryTerm ? 1 : 0;
    arguments.bandPixels = bandPixels;
    arguments.activeCoefficients = progress < shStarts[0] ? 1 : (progress < shStarts[1] ? 4 : SurfaceRendering::shCoefficients);
    return arguments;
}

std::expected<double, Error> SdfGrid::step(const Target& target, Training::Loss& loss) {
    if (stage + 1 < stageEdges.size() && getProgress() >= stageStarts[stage + 1]) {
        ++stage;
        if (std::expected<void, Error> refined = refine(stageEdges[stage]); !refined)
            return std::unexpected(std::move(refined.error()));
    }
    const Scenes::Camera& camera = target.view->camera;
    if (camera.width > imageWidth || camera.height > imageHeight)
        return fail(ErrorCode::InvalidArgument, "a train view is larger than the train views at open");
    SurfaceRendering::Arguments arguments = getArguments(camera, trainSamplesPerSide);
    arguments.seed = static_cast<std::uint32_t>(settings.seed * 0x9e3779b9u) ^ (iteration * 0x85ebca6bu);
    arguments.target = target.image;
    CommandList list(*device);
    for (const GridParameters* parameters : {&sdf, &colour, &environment})
        recordClear(list, parameters->getGradientPtr(), parameters->values.getSize());
    const DispatchSize pixels{camera.width, camera.height};
    list.dispatch(renderPipeline, pixels, arguments);
    loss.record(list, image.getPtr(), target.image, imageGradient.getPtr(), camera.width, camera.height);
    list.dispatch(backwardPipeline, pixels, arguments);
    SurfaceRendering::Arguments smooth = arguments;
    smooth.threads = getThreads(grid.size.w);
    list.dispatch(smoothPipeline, {smooth.threads}, smooth);
    ++adamStep;
    const float decay = std::pow(finalLearningRateFactor, getProgress()) * settings.learningRateScale;
    adam.record(list, sdf.getGroup(sdfLearningRate * grid.origin.w * decay), adamStep);
    adam.record(list, colour.getGroup(colourLearningRate * decay), adamStep);
    if (!cropping)
        adam.record(list, environment.getGroup(environmentLearningRate * decay), adamStep);
    recordRedistance(list);
    loss.recordReadback(list);
    if (std::expected<void, Error> submitted = submit(std::move(list)); !submitted)
        return std::unexpected(std::move(submitted.error()));
    ++iteration;
    return loss.read();
}

std::expected<void, Error> SdfGrid::render(const Scenes::Camera& camera, Core::Image& result) {
    if (camera.width > imageWidth || camera.height > imageHeight)
        if (std::expected<void, Error> opened = openImage(std::max(camera.width, imageWidth), std::max(camera.height, imageHeight)); !opened)
            return opened;
    const SurfaceRendering::Arguments arguments = getArguments(camera, renderSamplesPerSide);
    CommandList list(*device);
    list.dispatch(renderPipeline, {camera.width, camera.height}, arguments);
    list.download(image, imageReadback);
    if (std::expected<void, Error> submitted = submit(std::move(list)); !submitted)
        return submitted;
    std::expected<std::span<const float>, Error> values = imageReadback.read();
    if (!values)
        return std::unexpected(std::move(values.error()));
    std::ranges::copy(values->first(std::size_t{camera.width} * camera.height * 3), result.getPixels().begin());
    return {};
}

Statistics SdfGrid::getStatistics() const {
    const std::size_t floats = sdf.values.getSize() + colour.values.getSize() + environment.values.getSize();
    return {grid.size.w, floats * sizeof(float)};
}

void SdfGrid::close() {
    if (device && sdf.values)
        if (std::expected<std::vector<float>, Error> values = downloadSdf(); values) {
            closedSdf = std::move(*values);
            closedGrid = grid;
        }
    sdf = {};
    colour = {};
    environment = {};
    image = {};
    imageGradient = {};
    imageReadback = {};
    renderPipeline = {};
    backwardPipeline = {};
    seedPipeline = {};
    floodPipeline = {};
    redistancePipeline = {};
    seeds = {};
    smoothPipeline = {};
    clearPipeline = {};
    resamplePipeline = {};
    adam = {};
    imageWidth = 0;
    imageHeight = 0;
}

std::expected<void, std::string> SdfGrid::writeMesh(const std::filesystem::path& path) const {
    if (!closedSdf.empty())
        return writeIsoSurface(closedSdf, closedGrid, path);
    if (!sdf.values)
        return std::unexpected("the SDF grid has no trained SDF to write");
    std::expected<std::vector<float>, Error> values = downloadSdf();
    if (!values)
        return std::unexpected(values.error().message);
    return writeIsoSurface(*values, grid, path);
}

}
