#include "NoorRay.Reconstruction/Trainer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <chrono>
#include <format>
#include <random>
#include <utility>

#include "DiffRenderer/Scenes/Loading.h"
#include "NoorRay/Import/MeshBuilder.h"
#include "NoorRay.Reconstruction/Colmap.h"
#include "NoorRay.Reconstruction/Datasets.h"

namespace NoorRay::Reconstruction {

namespace {

using Clock = std::chrono::steady_clock;

//! While training, the viewport sees a new render at most this often, so the renders take little from the training.
constexpr std::chrono::milliseconds liveInterval{150};
constexpr int evaluateEvery = 1000;
constexpr std::size_t evaluatedViews = 3;
constexpr int statisticsEvery = 100;

double getMilliseconds(const Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

Scene::GaussianCloud toCloud(const DiffRenderer::Methods::TrainedGaussians& trained) {
    constexpr std::size_t trainedCoefficients = 16;
    const std::size_t count = trained.opacities.size();
    const std::size_t coefficients = Scene::getShCoefficientCount(trained.shDegree);
    Scene::GaussianCloud cloud;
    cloud.shDegree = trained.shDegree;
    cloud.positions.resize(count);
    cloud.scales.resize(count);
    cloud.rotations.resize(count);
    cloud.opacities = trained.opacities;
    cloud.sh.resize(count * coefficients * 3);
    for (std::size_t index = 0; index < count; ++index) {
        const float* mean = trained.means.data() + index * 3;
        const float* scale = trained.scales.data() + index * 3;
        const float* quaternion = trained.quaternions.data() + index * 4;
        const float* source = trained.sh.data() + index * trainedCoefficients * 3;
        std::uint16_t* sh = cloud.sh.data() + index * coefficients * 3;
        cloud.positions[index] = {mean[0], mean[1], mean[2]};
        cloud.scales[index] = {scale[0], scale[1], scale[2]};
        cloud.rotations[index] = {quaternion[0], quaternion[1], quaternion[2], quaternion[3]};
        for (std::size_t value = 0; value < coefficients * 3; ++value)
            sh[value] = Core::toHalf(source[value]).bits;
    }
    return cloud;
}

std::uint8_t toByte(const float value) {
    return static_cast<std::uint8_t>(std::lround(std::clamp(value, 0.0f, 1.0f) * 255.0f));
}

float toSrgb(const float value) {
    return value <= 0.0031308f ? value * 12.92f : 1.055f * std::pow(value, 1.0f / 2.4f) - 0.055f;
}

//! The occlusion of nvdiffrec darkens all of the shading; the Principled BSDF has no input for it, so it darkens the base colour.
Core::float3 getBaseColor(const float* kd, const float occlusion) {
    const float scale = 1.0f - occlusion;
    return {kd[0] * scale, kd[1] * scale, kd[2] * scale};
}

Scene::Texture makeTexture(std::string name, const NoorRHI::ImageFormat format, const std::uint32_t width, const std::uint32_t height,
                           const std::vector<std::array<std::uint8_t, 4>>& pixels) {
    Scene::Texture texture{.name = std::move(name), .format = format, .width = width, .height = height};
    texture.data.resize(pixels.size() * 4);
    std::memcpy(texture.data.data(), pixels.data(), texture.data.size());
    texture.levels = {{.offset = 0, .size = texture.data.size(), .width = width, .height = height}};
    return texture;
}

std::vector<Scene::Texture> toTextures(const DiffRenderer::Methods::TrainedMesh& trained) {
    const std::size_t count = std::size_t{trained.textureWidth} * trained.textureHeight;
    std::vector<std::array<std::uint8_t, 4>> base(count);
    std::vector<std::array<std::uint8_t, 4>> surface(count);
    std::vector<std::array<std::uint8_t, 4>> normal(count);
    for (std::size_t texel = 0; texel < count; ++texel) {
        const float* value = trained.texels.data() + texel * 9;
        const Core::float3 color = getBaseColor(value, value[3]);
        base[texel] = {toByte(toSrgb(color.x)), toByte(toSrgb(color.y)), toByte(toSrgb(color.z)), 255};
        surface[texel] = {toByte(value[3]), toByte(value[4]), toByte(value[5]), 255};
        normal[texel] = {toByte(value[6] * 0.5f + 0.5f), toByte(value[7] * 0.5f + 0.5f), toByte(value[8] * 0.5f + 0.5f), 255};
    }
    const std::uint32_t width = trained.textureWidth;
    const std::uint32_t height = trained.textureHeight;
    return {makeTexture("kd", NoorRHI::ImageFormat::Rgba8Srgb, width, height, base), makeTexture("ks", NoorRHI::ImageFormat::Rgba8Unorm, width, height, surface),
            makeTexture("normal", NoorRHI::ImageFormat::Rgba8Unorm, width, height, normal)};
}

//! A textured mesh gets a vertex for each UV vertex, so the UVs split at the seams; the normals come from the unsplit mesh, so the shading does not.
TrainerMesh toMesh(const DiffRenderer::Methods::TrainedMesh& trained) {
    const std::size_t vertices = trained.positions.size() / 3;
    std::vector<Core::float3> positions(vertices);
    for (std::size_t vertex = 0; vertex < vertices; ++vertex)
        positions[vertex] = {trained.positions[vertex * 3], trained.positions[vertex * 3 + 1], trained.positions[vertex * 3 + 2]};
    const std::vector<Core::float3> normals = Import::MeshBuilder::computeNormals(positions, trained.triangles);
    Import::MeshData data;
    TrainerMesh result;
    if (trained.texels.empty()) {
        data.positions = positions;
        data.normals = normals;
        data.indices = trained.triangles;
        data.colors.resize(vertices);
        float roughness = 0.0f;
        float metallic = 0.0f;
        for (std::size_t vertex = 0; vertex < vertices; ++vertex) {
            const float* value = trained.vertexMaterials.data() + vertex * 6;
            const Core::float3 color = getBaseColor(value, value[3]);
            data.colors[vertex] = {color.x, color.y, color.z, 1.0f};
            roughness += value[4];
            metallic += value[5];
        }
        result.roughness = vertices > 0 ? roughness / static_cast<float>(vertices) : 0.5f;
        result.metallic = vertices > 0 ? metallic / static_cast<float>(vertices) : 0.0f;
    } else {
        const std::size_t split = trained.uvs.size() / 2;
        data.positions.resize(split);
        data.normals.resize(split);
        data.uvs[0].resize(split);
        for (std::size_t corner = 0; corner < trained.triangles.size(); ++corner) {
            const std::uint32_t target = trained.uvTriangles[corner];
            data.positions[target] = positions[trained.triangles[corner]];
            data.normals[target] = normals[trained.triangles[corner]];
            data.uvs[0][target] = {trained.uvs[target * 2], 1.0f - trained.uvs[target * 2 + 1]};
        }
        data.indices = trained.uvTriangles;
        result.textures = toTextures(trained);
    }
    result.mesh = Import::MeshBuilder::build(std::move(data));
    return result;
}

}

void Trainer::open(NoorRHI::Device& target, const std::filesystem::path& photos, std::string chosenMethod, const int iterations) {
    close();
    device = &target;
    {
        std::lock_guard lock(mutex);
        status = {.stage = Stage::Colmap, .message = "Starting COLMAP"};
        wantedMethod = std::move(chosenMethod);
        wantedIterations = iterations;
        ++methodRequest;
        wantTraining = false;
        frameRequest.reset();
        frame.reset();
        result.reset();
    }
    catalog = std::make_unique<DiffRenderer::Methods::Catalog>();
    worker = std::jthread([this, photos](std::stop_token token) { run(std::move(token), photos); });
}

void Trainer::close() {
    if (worker.joinable()) {
        worker.request_stop();
        worker.join();
    }
    catalog.reset();
    std::lock_guard lock(mutex);
    status = {};
    dataset.reset();
    frameRequest.reset();
    frame.reset();
    result.reset();
}

void Trainer::setMethod(std::string chosenMethod, const int iterations) {
    std::lock_guard lock(mutex);
    wantedMethod = std::move(chosenMethod);
    wantedIterations = iterations;
    ++methodRequest;
    wake.notify_all();
}

void Trainer::setTraining(const bool training) {
    std::lock_guard lock(mutex);
    wantTraining = training;
    status.training = training && status.iteration < status.iterations;
    wake.notify_all();
}

void Trainer::requestFrame(const DiffRenderer::Scenes::Camera& camera) {
    std::lock_guard lock(mutex);
    frameRequest = camera;
    wake.notify_all();
}

std::optional<TrainerFrame> Trainer::takeFrame() {
    std::lock_guard lock(mutex);
    return std::exchange(frame, std::nullopt);
}

void Trainer::requestResult() {
    std::lock_guard lock(mutex);
    status.readingResult = true;
    wake.notify_all();
}

std::optional<TrainerResult> Trainer::takeResult() {
    std::lock_guard lock(mutex);
    return std::exchange(result, std::nullopt);
}

TrainerStatus Trainer::getStatus() {
    std::lock_guard lock(mutex);
    return status;
}

const DiffRenderer::Scenes::Dataset* Trainer::getDataset() const {
    std::lock_guard lock(mutex);
    return status.stage == Stage::Ready ? dataset.get() : nullptr;
}

void Trainer::setStage(const Stage stage, std::string message, const float progress) {
    std::lock_guard lock(mutex);
    status.stage = stage;
    status.message = std::move(message);
    status.progress = progress;
}

void Trainer::fail(std::string message) {
    setStage(Stage::Failed, std::move(message), 0.0f);
}

std::expected<DiffRenderer::Scenes::Dataset, std::string> Trainer::loadFolder(const std::stop_token& token, const std::filesystem::path& folder) {
    if (detectDataset(folder) == DatasetKind::Synthetic) {
        setStage(Stage::Loading, "Loading the renders", 0.0f);
        return DiffRenderer::Scenes::loadNerfSynthetic(folder.filename().string(), folder, {});
    }
    std::expected<ColmapDataset, Core::Error> colmap =
        runColmap(folder, token, [this](const ColmapProgress& progress) { setStage(Stage::Colmap, progress.stage, progress.fraction); });
    if (!colmap)
        return std::unexpected(std::format("COLMAP failed: {}", colmap.error().message));
    setStage(Stage::Loading, "Loading the photographs", 0.0f);
    return DiffRenderer::Scenes::loadColmap(colmap->sparse, colmap->images, {});
}

bool Trainer::load(const std::stop_token token, const std::filesystem::path& photos) {
    std::expected<DiffRenderer::Scenes::Dataset, std::string> loaded = loadFolder(token, photos);
    if (!loaded) {
        fail(std::format("The dataset does not load: {}", loaded.error()));
        return false;
    }
    if (loaded->train.empty()) {
        fail("The dataset has no training views");
        return false;
    }
    loaded->id = "photos";
    loaded->name = photos.filename().string();
    loaded->kind = detectDataset(photos) == DatasetKind::Synthetic ? DiffRenderer::Scenes::Kind::Object : DiffRenderer::Scenes::Kind::Scene;
    auto opened = std::make_unique<DiffRenderer::Scenes::Dataset>(std::move(*loaded));
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    for (const DiffRenderer::Scenes::View& view : opened->train) {
        width = std::max(width, view.image.getWidth());
        height = std::max(height, view.image.getHeight());
    }
    targets.emplace();
    if (std::expected<void, NoorRHI::Error> uploaded = targets->open(*device, *opened); !uploaded) {
        fail(std::format("The photographs do not fit on the GPU: {}", uploaded.error().message));
        return false;
    }
    loss.emplace();
    if (std::expected<void, NoorRHI::Error> created = loss->open(*device, width, height, DiffRenderer::Training::LossKind::L1Ssim); !created) {
        fail(created.error().message);
        return false;
    }
    std::lock_guard lock(mutex);
    dataset = std::move(opened);
    status.stage = Stage::Ready;
    status.message = std::format("{} photographs, {} for the test", dataset->train.size() + dataset->test.size(), dataset->test.size());
    status.progress = 1.0f;
    return true;
}

bool Trainer::openMethod(const std::string& id, const int iterations) {
    if (method)
        method->close();
    method = catalog->find(id);
    iteration = 0;
    order.clear();
    {
        std::lock_guard lock(mutex);
        status.method = id;
        status.iteration = 0;
        status.iterations = iterations;
        status.loss = 0.0;
        status.psnr = 0.0;
        status.primitives = 0;
        status.hasResult = false;
        status.resultIsMesh = false;
        status.training = wantTraining;
    }
    if (!method || !method->supports(dataset->kind)) {
        method = nullptr;
        setStage(Stage::Ready, std::format("The method {} does not train on photographs", id), 1.0f);
        return false;
    }
    DiffRenderer::Methods::Settings settings;
    settings.iterations = iterations;
    if (std::expected<void, NoorRHI::Error> opened = method->open(*device, *dataset, settings); !opened) {
        method = nullptr;
        setStage(Stage::Ready, std::format("The method {} does not open: {}", id, opened.error().message), 1.0f);
        return false;
    }
    const std::expected<std::optional<DiffRenderer::Methods::TrainedGaussians>, NoorRHI::Error> initial = method->readGaussians();
    if (!initial)
        setStage(Stage::Ready, std::format("The Gaussians do not read back: {}", initial.error().message), 1.0f);
    bool hasMesh = false;
    if (initial && !*initial) {
        const std::expected<std::optional<DiffRenderer::Methods::TrainedMesh>, NoorRHI::Error> mesh = method->readMesh();
        hasMesh = mesh && mesh->has_value();
    }
    std::lock_guard lock(mutex);
    status.primitives = method->getStatistics().primitives;
    status.hasResult = (initial && initial->has_value()) || hasMesh;
    status.resultIsMesh = hasMesh;
    return true;
}

bool Trainer::step(const int iterations) {
    const auto count = static_cast<std::uint32_t>(dataset->train.size());
    if (iteration % static_cast<int>(count) == 0) {
        order.resize(count);
        for (std::uint32_t index = 0; index < count; ++index)
            order[index] = index;
        std::mt19937_64 random(static_cast<std::uint64_t>(iteration / static_cast<int>(count)) * 0x9e3779b97f4a7c15ull + 1);
        std::ranges::shuffle(order, random);
    }
    const Clock::time_point start = Clock::now();
    std::expected<double, NoorRHI::Error> stepped = method->step(targets->get(order[static_cast<std::size_t>(iteration) % count]), *loss);
    if (!stepped) {
        setStage(Stage::Ready, std::format("The training failed at iteration {}: {}", iteration, stepped.error().message), 1.0f);
        method = nullptr;
        std::lock_guard lock(mutex);
        status.hasResult = false;
        return false;
    }
    ++iteration;
    const double milliseconds = getMilliseconds(start);
    const bool last = iteration >= iterations;
    if (last || iteration % evaluateEvery == 0)
        evaluate();
    std::lock_guard lock(mutex);
    status.iteration = iteration;
    status.loss = *stepped;
    status.stepMilliseconds = status.stepMilliseconds == 0.0 ? milliseconds : 0.95 * status.stepMilliseconds + 0.05 * milliseconds;
    if (iteration % statisticsEvery == 0 || last)
        status.primitives = method->getStatistics().primitives;
    if (last) {
        wantTraining = false;
        status.training = false;
    }
    return true;
}

void Trainer::evaluate() {
    double psnr = 0.0;
    std::size_t count = 0;
    const std::size_t stride = std::max<std::size_t>(dataset->test.size() / evaluatedViews, 1);
    for (std::size_t index = 0; index < dataset->test.size() && count < evaluatedViews; index += stride) {
        const DiffRenderer::Scenes::View& view = dataset->test[index];
        DiffRenderer::Core::Image image(view.camera.width, view.camera.height);
        if (!method->render(view.camera, image))
            return;
        image.setEncoding(view.image.getEncoding());
        psnr += DiffRenderer::Core::getPsnr(DiffRenderer::Core::getMeanSquaredError(image, view.image));
        ++count;
    }
    if (count == 0)
        return;
    std::lock_guard lock(mutex);
    status.psnr = psnr / static_cast<double>(count);
}

void Trainer::readResult() {
    std::expected<std::optional<DiffRenderer::Methods::TrainedGaussians>, NoorRHI::Error> trained;
    std::expected<std::optional<DiffRenderer::Methods::TrainedMesh>, NoorRHI::Error> mesh;
    if (method && iteration > 0) {
        trained = method->readGaussians();
        if (trained && !*trained)
            mesh = method->readMesh();
    }
    const bool read = trained && mesh;
    if (!read)
        setStage(Stage::Ready, std::format("The result does not read back: {}", (trained ? mesh.error() : trained.error()).message), 1.0f);
    std::optional<TrainerResult> converted;
    if (read && *trained)
        converted = TrainerResult{toCloud(**trained), iteration};
    else if (read && *mesh)
        converted = TrainerResult{toMesh(**mesh), iteration};
    std::lock_guard lock(mutex);
    status.readingResult = false;
    if (!read)
        status.hasResult = false;
    if (converted)
        result = std::move(converted);
}

bool Trainer::render(const DiffRenderer::Scenes::Camera& requested) {
    if (!method)
        return false;
    const DiffRenderer::Scenes::View& largest = dataset->train.front();
    const float fit = std::min({1.0f, static_cast<float>(largest.image.getWidth()) / static_cast<float>(requested.width),
                                static_cast<float>(largest.image.getHeight()) / static_cast<float>(requested.height)});
    DiffRenderer::Scenes::Camera camera = requested;
    camera.width = std::max(1u, static_cast<std::uint32_t>(static_cast<float>(requested.width) * fit));
    camera.height = std::max(1u, static_cast<std::uint32_t>(static_cast<float>(requested.height) * fit));
    camera.intrinsics = {requested.intrinsics.x * fit, requested.intrinsics.y * fit, requested.intrinsics.z * fit, requested.intrinsics.w * fit};
    if (rendered.getWidth() != camera.width || rendered.getHeight() != camera.height)
        rendered = DiffRenderer::Core::Image(camera.width, camera.height);
    if (std::expected<void, NoorRHI::Error> drawn = method->render(camera, rendered); !drawn) {
        setStage(Stage::Ready, std::format("The render failed: {}", drawn.error().message), 1.0f);
        return false;
    }
    rendered.setEncoding(largest.image.getEncoding());
    TrainerFrame converted{camera.width, camera.height, std::vector<std::uint32_t>(std::size_t{camera.width} * camera.height)};
    DiffRenderer::Core::toRgba8(rendered, converted.pixels);
    std::lock_guard lock(mutex);
    frame = std::move(converted);
    return true;
}

void Trainer::run(const std::stop_token token, const std::filesystem::path photos) {
    if (!load(token, photos)) {
        targets.reset();
        loss.reset();
        return;
    }
    std::uint64_t openedRequest = 0;
    int iterations = 0;
    std::optional<DiffRenderer::Scenes::Camera> lastCamera;
    Clock::time_point lastFrame = Clock::now();
    while (!token.stop_requested()) {
        std::string id;
        std::uint64_t request = 0;
        bool training = false;
        bool reading = false;
        std::optional<DiffRenderer::Scenes::Camera> camera;
        {
            std::unique_lock lock(mutex);
            wake.wait(lock, token, [&] {
                return methodRequest != openedRequest || frameRequest || status.readingResult || (wantTraining && method && iteration < status.iterations);
            });
            if (token.stop_requested())
                break;
            id = wantedMethod;
            iterations = wantedIterations;
            request = methodRequest;
            training = wantTraining;
            reading = status.readingResult;
            camera = std::exchange(frameRequest, std::nullopt);
        }
        if (request != openedRequest) {
            openedRequest = request;
            openMethod(id, iterations);
            if (!camera)
                camera = lastCamera;
        }
        if (reading)
            readResult();
        if (camera) {
            lastCamera = camera;
            render(*camera);
            lastFrame = Clock::now();
            continue;
        }
        if (!training || !method || iteration >= iterations)
            continue;
        if (!step(iterations))
            continue;
        if (lastCamera && (Clock::now() - lastFrame >= liveInterval || iteration >= iterations)) {
            render(*lastCamera);
            lastFrame = Clock::now();
        }
    }
    if (method)
        method->close();
    method = nullptr;
    targets.reset();
    loss.reset();
}

}
