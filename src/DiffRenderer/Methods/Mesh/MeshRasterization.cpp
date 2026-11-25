#include "DiffRenderer/Methods/Mesh/MeshRasterization.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <print>
#include <random>
#include <utility>

#include "DiffRenderer/Methods/Mesh/Remesher.h"
#include "DiffRenderer/Methods/Mesh/VisualHull.h"

namespace DiffRenderer::Methods {

using namespace NoorRHI;

namespace {

constexpr int hullResolution = 160;
constexpr int sphereSubdivisions = 4;
constexpr float startEdgesAcross = 64.0f;
constexpr int startPasses = 6;
constexpr int remeshPasses = 3;
constexpr int remeshCount = 20;
//! nvdiffrec gives each pass the same iterations; the second keeps the topology.
constexpr float remeshEnd = 0.5f;
constexpr float refineEnd = 0.4f;
constexpr float finestFraction = 1.0f / 12.0f;
constexpr float maxTriangles = 400'000.0f;
constexpr float positionRate = 1.0f;
//! The learning rates of the materials and the light in the two passes of the nerf configurations of nvdiffrec.
constexpr std::array<float, 2> materialRates{0.03f, 0.01f};
constexpr int textureWarmup = 100;
constexpr float finalDecay = 0.1f;
constexpr std::uint32_t lightSize = 512;
constexpr float smoothWeight = 0.03f;
constexpr int smoothRamp = 500;
constexpr float occlusionWeight = 0.001f;
constexpr std::uint32_t materialSeed = 1;
constexpr std::uint32_t textureResolution = 2048;
constexpr float nearFraction = 0.01f;
constexpr float farFraction = 100.0f;

constexpr Description description{
    .id = "mesh",
    .name = "Mesh + PBR (nvdiffrec)",
    .representation = "triangle mesh with PBR materials on each vertex, then kd, ks and normal textures; a learned environment light",
    .renderer = "hardware rasterizer with analytic antialiasing (nvdiffrast), split-sum image based lighting (nvdiffrec)",
    .discontinuities = "analytic antialiasing of the silhouette edges",
    .reference = "Laine et al. 2020; Nicolet et al. 2021; Munkberg et al. 2022",
    .scene = true,
    .object = true,
    .volume = false,
    .hasBoundaryTerm = true,
};

std::expected<void, Error> submit(Device& device, CommandList&& list) {
    std::expected<Token, Error> token = device.submit(std::move(list));
    if (!token)
        return std::unexpected(std::move(token.error()));
    return device.wait(*token);
}

SurfaceMesh makeStartSphere(const Scenes::Dataset& dataset) {
    const Vector3 center{dataset.center.x, dataset.center.y, dataset.center.z};
    std::vector<float> distances;
    distances.reserve(dataset.points.positions.size());
    for (const float3 point : dataset.points.positions)
        distances.push_back((Vector3{point.x, point.y, point.z} - center).getLength());
    float radius = 0.25f * dataset.extent;
    if (!distances.empty()) {
        const auto middle = distances.begin() + static_cast<std::ptrdiff_t>(distances.size() / 2);
        std::ranges::nth_element(distances, middle);
        radius = *middle;
    }
    return makeIcosphere(center, radius, sphereSubdivisions);
}

//! The random start of the materials of nvdiffrec: kd and the roughness and metallic values uniform in their ranges, the occlusion below 0.01.
void randomizeMaterials(SurfaceMesh& mesh) {
    std::mt19937 random(materialSeed);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);
    for (std::size_t vertex = 0; vertex < mesh.getVertexCount(); ++vertex) {
        mesh.colors[vertex] = {unit(random), unit(random), unit(random)};
        const float occlusion = 0.01f * unit(random);
        const float roughness = EnvironmentLight::minimumRoughness + (1.0f - EnvironmentLight::minimumRoughness) * unit(random);
        mesh.materials[vertex] = {occlusion, roughness, unit(random)};
    }
}

float getDiagonal(const SurfaceMesh& mesh) {
    Vector3 low{1e30f, 1e30f, 1e30f};
    Vector3 high{-1e30f, -1e30f, -1e30f};
    for (const Vector3 point : mesh.positions) {
        low = {std::min(low.x, point.x), std::min(low.y, point.y), std::min(low.z, point.z)};
        high = {std::max(high.x, point.x), std::max(high.y, point.y), std::max(high.z, point.z)};
    }
    return (high - low).getLength();
}

}

const Description& MeshRasterization::getDescription() const {
    return description;
}

std::expected<void, Error> MeshRasterization::open(Device& target, const Scenes::Dataset& dataset, const Settings& chosen) {
    device = &target;
    settings = chosen;
    iteration = 0;
    textureStart = -1;
    lightBuilt = false;
    atlas = {};
    texture.clear();
    std::optional<SurfaceMesh> hull = carveVisualHull(dataset, hullResolution);
    mesh = hull ? std::move(*hull) : makeStartSphere(dataset);
    startLength = getDiagonal(mesh) / startEdgesAcross;
    Remesher(mesh).remesh(startLength, startPasses);
    randomizeMaterials(mesh);
    area = mesh.getArea();
    std::size_t maxPixels = 0;
    for (const std::vector<Scenes::View>* views : {&dataset.train, &dataset.test})
        for (const Scenes::View& view : *views)
            maxPixels = std::max(maxPixels, std::size_t{view.camera.width} * view.camera.height);
    if (std::expected<void, Error> opened =
            rasterizer.open(target, maxPixels, dataset.background, settings.boundaryTerm, nearFraction * dataset.extent, farFraction * dataset.extent);
        !opened)
        return opened;
    if (std::expected<void, Error> opened = optimizer.open(target, {}); !opened)
        return opened;
    if (std::expected<void, Error> opened = materialAdam.open(target, {.epsilon = 1e-8f}); !opened)
        return opened;
    if (std::expected<void, Error> opened = light.open(target, lightSize); !opened)
        return opened;
    rasterizer.setLight(light.getLight());
    return setMesh();
}

std::expected<void, Error> MeshRasterization::setMesh() {
    if (std::expected<void, Error> uploaded = rasterizer.upload(mesh); !uploaded)
        return uploaded;
    if (std::expected<void, Error> set = optimizer.setMesh(mesh); !set)
        return set;
    return restartMaterialMoments(mesh.getVertexCount() * MeshRasterizing::vertexChannels);
}

std::expected<void, Error> MeshRasterization::restartMaterialMoments(const std::size_t count) {
    const std::vector<float> zeros(count, 0.0f);
    for (std::expected<void, Error> step :
         {materialFirst.open(*device, count), materialSecond.open(*device, count), materialFirst.upload(zeros), materialSecond.upload(zeros)})
        if (!step)
            return step;
    materialStep = 0;
    return {};
}

float MeshRasterization::getSchedule() const {
    const int lastRemesh = static_cast<int>(remeshEnd * static_cast<float>(settings.iterations));
    const bool textured = textureStart >= 0;
    const int start = textured ? textureStart : 0;
    const int warmup = textured ? textureWarmup : 0;
    const int length = std::max(1, (textured ? settings.iterations : lastRemesh) - start - warmup);
    const int passIteration = iteration - start;
    if (passIteration < warmup)
        return static_cast<float>(passIteration) / static_cast<float>(warmup);
    return std::pow(finalDecay, static_cast<float>(passIteration - warmup) / static_cast<float>(length));
}

float MeshRasterization::getTargetLength() const {
    const float progress = std::min(1.0f, static_cast<float>(iteration) / (refineEnd * static_cast<float>(settings.iterations)));
    const float scheduled = startLength * std::pow(finestFraction, progress);
    const float budget = std::sqrt(area / (maxTriangles * std::sqrt(3.0f) / 4.0f));
    return std::max(scheduled, budget);
}

std::expected<void, Error> MeshRasterization::remesh() {
    if (std::expected<void, Error> downloaded = rasterizer.download(mesh); !downloaded)
        return downloaded;
    area = mesh.getArea();
    Remesher(mesh).remesh(getTargetLength(), remeshPasses);
    return setMesh();
}

std::expected<void, Error> MeshRasterization::startTexture() {
    std::expected<TextureAtlas, std::string> unwrapped = unwrapMesh(mesh, textureResolution);
    if (!unwrapped)
        return fail(ErrorCode::InvalidState, std::move(unwrapped.error()));
    atlas = std::move(*unwrapped);
    const std::vector<float> kd = bakeVertexValues(mesh, atlas, mesh.colors);
    const std::vector<float> ks = bakeVertexValues(mesh, atlas, mesh.materials);
    std::vector<float> texels(atlas.getTexelCount() * MeshRasterizing::texelChannels);
    for (std::size_t texel = 0; texel < atlas.getTexelCount(); ++texel) {
        float* value = texels.data() + texel * MeshRasterizing::texelChannels;
        std::copy_n(kd.begin() + static_cast<std::ptrdiff_t>(texel * 3), 3, value);
        std::copy_n(ks.begin() + static_cast<std::ptrdiff_t>(texel * 3), 3, value + 3);
        std::ranges::copy(std::array{0.0f, 0.0f, 1.0f}, value + 6);
    }
    if (std::expected<void, Error> set = rasterizer.setTexture(atlas, texels); !set)
        return set;
    textureStart = iteration;
    if (std::expected<void, Error> restarted = light.restartMoments(); !restarted)
        return restarted;
    return restartMaterialMoments(texels.size());
}

std::expected<double, Error> MeshRasterization::step(const Target& target, Training::Loss& loss) {
    const Scenes::Camera& camera = target.view->camera;
    const bool textured = textureStart >= 0;
    const auto pixels = static_cast<float>(std::size_t{camera.width} * camera.height);
    const int passIteration = iteration - (textured ? textureStart : 0);
    const float smoothRampFactor = std::min(1.0f, static_cast<float>(passIteration) / static_cast<float>(smoothRamp));
    rasterizer.setRegularizers(static_cast<std::uint32_t>(iteration), smoothWeight * smoothRampFactor / pixels, occlusionWeight / pixels);
    CommandList list(*device);
    light.recordBuild(list);
    light.recordClearGradients(list);
    if (std::expected<void, Error> recorded = rasterizer.recordForward(list, camera); !recorded)
        return std::unexpected(std::move(recorded.error()));
    loss.record(list, rasterizer.getImage(), target.image, rasterizer.getImageGradient(), camera.width, camera.height);
    rasterizer.recordBackward(list);
    light.recordBackward(list);
    const float schedule = getSchedule();
    const float materialRate = materialRates[textured ? 1 : 0] * schedule * settings.learningRateScale;
    materialAdam.record(list,
                        {textured ? rasterizer.getTexels() : rasterizer.getVertexMaterials(),
                         textured ? rasterizer.getTexelGradient() : rasterizer.getVertexMaterialGradient(), materialFirst.getPtr(), materialSecond.getPtr(),
                         textured ? rasterizer.getTexelValueCount() : rasterizer.getVertexCount() * MeshRasterizing::vertexChannels, materialRate},
                        ++materialStep);
    rasterizer.recordClamp(list);
    light.recordStep(list, materialAdam, materialRate);
    lightBuilt = false;
    optimizer.record(list, rasterizer.getPositions(), rasterizer.getPositionGradient(), positionRate * schedule * getTargetLength() * settings.learningRateScale);
    loss.recordReadback(list);
    if (std::expected<void, Error> submitted = submit(*device, std::move(list)); !submitted)
        return std::unexpected(std::move(submitted.error()));
    std::expected<double, Error> value = loss.read();
    if (!value)
        return value;
    ++iteration;
    const int lastRemesh = static_cast<int>(remeshEnd * static_cast<float>(settings.iterations));
    const int interval = std::max(1, lastRemesh / remeshCount);
    if (iteration % interval != 0 || iteration > lastRemesh)
        return value;
    atlas = {};
    if (std::expected<void, Error> remeshed = remesh(); !remeshed)
        return std::unexpected(std::move(remeshed.error()));
    if (iteration + interval <= lastRemesh)
        return value;
    if (std::expected<void, Error> textured = startTexture(); !textured)
        return std::unexpected(std::move(textured.error()));
    return value;
}

std::expected<void, Error> MeshRasterization::render(const Scenes::Camera& camera, Core::Image& image) {
    if (!lightBuilt) {
        CommandList list(*device);
        light.recordBuild(list);
        if (std::expected<void, Error> submitted = submit(*device, std::move(list)); !submitted)
            return submitted;
        lightBuilt = true;
    }
    return rasterizer.render(camera, image);
}

Statistics MeshRasterization::getStatistics() const {
    const std::size_t vertices = rasterizer.getVertexCount();
    const std::size_t triangles = rasterizer.getTriangleCount();
    const std::size_t textureBytes =
        std::size_t{rasterizer.getTexelValueCount()} * sizeof(float) + atlas.uvs.size() * sizeof(float2) + atlas.uvTriangles.size() * sizeof(std::uint32_t);
    const std::size_t vertexBytes = vertices * (3 + (rasterizer.getTexelValueCount() > 0 ? 0 : MeshRasterizing::vertexChannels)) * sizeof(float);
    return {triangles, vertexBytes + triangles * 6 * sizeof(std::uint32_t) + textureBytes + light.getByteCount()};
}

void MeshRasterization::close() {
    if (device && rasterizer.getVertexCount() > 0)
        if (std::expected<void, Error> downloaded = download(mesh, texture); !downloaded)
            std::println(stderr, "the trained mesh does not download: {}", downloaded.error().message);
    rasterizer.close();
    optimizer.close();
    light.close();
    materialAdam = {};
    materialFirst = {};
    materialSecond = {};
    device = nullptr;
}

std::expected<std::optional<TrainedMesh>, Error> MeshRasterization::readMesh() {
    if (!device || rasterizer.getVertexCount() == 0)
        return std::nullopt;
    SurfaceMesh trained;
    std::vector<float> texels;
    if (std::expected<void, Error> downloaded = download(trained, texels); !downloaded)
        return std::unexpected(std::move(downloaded.error()));
    TrainedMesh result;
    result.positions.reserve(trained.getVertexCount() * 3);
    for (const Vector3 position : trained.positions)
        result.positions.insert(result.positions.end(), {position.x, position.y, position.z});
    result.triangles = std::move(trained.triangles);
    if (texels.empty()) {
        result.vertexMaterials.reserve(trained.getVertexCount() * MeshRasterizing::vertexChannels);
        for (std::size_t vertex = 0; vertex < trained.getVertexCount(); ++vertex) {
            const Vector3 kd = trained.colors[vertex];
            const Vector3 ks = trained.materials[vertex];
            result.vertexMaterials.insert(result.vertexMaterials.end(), {kd.x, kd.y, kd.z, ks.x, ks.y, ks.z});
        }
        return result;
    }
    result.uvs.reserve(atlas.uvs.size() * 2);
    for (const float2 uv : atlas.uvs)
        result.uvs.insert(result.uvs.end(), {uv.x, uv.y});
    result.uvTriangles = atlas.uvTriangles;
    result.textureWidth = atlas.width;
    result.textureHeight = atlas.height;
    result.texels = std::move(texels);
    return result;
}

std::expected<void, Error> MeshRasterization::download(SurfaceMesh& trained, std::vector<float>& texels) const {
    if (std::expected<void, Error> downloaded = rasterizer.download(trained); !downloaded)
        return downloaded;
    texels.resize(atlas.getTexelCount() * MeshRasterizing::texelChannels);
    if (texels.empty())
        return {};
    return rasterizer.downloadTexture(texels);
}

std::expected<void, std::string> MeshRasterization::writeMesh(const std::filesystem::path& path) const {
    if (!device)
        return writeTrainedMesh(mesh, texture, path);
    SurfaceMesh trained = mesh;
    std::vector<float> texels;
    if (std::expected<void, Error> downloaded = download(trained, texels); !downloaded)
        return std::unexpected(std::move(downloaded.error().message));
    return writeTrainedMesh(trained, texels, path);
}

std::expected<void, std::string> MeshRasterization::writeTrainedMesh(const SurfaceMesh& trained, const std::span<const float> texels,
                                                                     const std::filesystem::path& path) const {
    if (texels.empty())
        return std::unexpected("the mesh has no trained texture to write");
    return saveTexturedObj(trained, atlas, texels, path);
}

}
