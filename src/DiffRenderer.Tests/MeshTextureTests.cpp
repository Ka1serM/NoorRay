#include <cmath>
#include <format>
#include <print>
#include <vector>

#include "DiffRenderer.Tests/Tests.h"
#include "DiffRenderer/Core/Image.h"
#include "DiffRenderer/Methods/Mesh/MeshRasterizer.h"
#include "DiffRenderer/Methods/Mesh/TextureAtlas.h"
#include "DiffRenderer/Methods/Mesh/TrainableLight.h"
#include "DiffRenderer/Training/GpuAdam.h"
#include "DiffRenderer/Training/Loss.h"

namespace DiffRenderer::Tests {

using namespace NoorRHI;
using Methods::Vector3;

namespace {

constexpr std::uint32_t size = 96;
constexpr std::uint32_t textureResolution = 256;
constexpr int fitSteps = 300;
constexpr std::uint32_t lightSize = 64;

Scenes::Camera makeCamera() {
    Scenes::Camera camera;
    camera.worldToCamera = {{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 3}}};
    camera.intrinsics = {110.0f, 110.0f, 48.0f, 48.0f};
    camera.width = size;
    camera.height = size;
    return camera;
}

Methods::SurfaceMesh makeColoredSphere(const bool inverted) {
    Methods::SurfaceMesh mesh = Methods::makeIcosphere({}, 1.0f, 3);
    for (std::size_t vertex = 0; vertex < mesh.positions.size(); ++vertex) {
        const Vector3 color = mesh.positions[vertex] * 0.5f + Vector3{0.5f, 0.5f, 0.5f};
        mesh.colors[vertex] = inverted ? Vector3{1.0f, 1.0f, 1.0f} - color : color;
    }
    return mesh;
}

//! kd and ks of the vertices in the nine values of each texel, with a flat normal map.
std::vector<float> bakeMaterials(const Methods::SurfaceMesh& mesh, const Methods::TextureAtlas& atlas) {
    const std::vector<float> kd = Methods::bakeVertexValues(mesh, atlas, mesh.colors);
    const std::vector<float> ks = Methods::bakeVertexValues(mesh, atlas, mesh.materials);
    std::vector<float> texels(atlas.getTexelCount() * MeshRasterizing::texelChannels);
    for (std::size_t texel = 0; texel < atlas.getTexelCount(); ++texel)
        for (std::size_t channel = 0; channel < 3; ++channel) {
            texels[texel * MeshRasterizing::texelChannels + channel] = kd[texel * 3 + channel];
            texels[texel * MeshRasterizing::texelChannels + 3 + channel] = ks[texel * 3 + channel];
            texels[texel * MeshRasterizing::texelChannels + 6 + channel] = channel == 2 ? 1.0f : 0.0f;
        }
    return texels;
}

double getMeanDifference(const Core::Image& a, const Core::Image& b) {
    double sum = 0.0;
    for (std::size_t index = 0; index < a.getPixels().size(); ++index)
        sum += std::abs(a.getPixels()[index] - b.getPixels()[index]);
    return sum / static_cast<double>(a.getPixels().size());
}

}

Result testMeshTexture(Device& device) {
    const Methods::SurfaceMesh mesh = makeColoredSphere(false);
    std::expected<Methods::TextureAtlas, std::string> atlas = Methods::unwrapMesh(mesh, textureResolution);
    if (!atlas)
        return std::unexpected(atlas.error());
    const std::vector<float> texels = bakeMaterials(mesh, *atlas);
    const Scenes::Camera camera = makeCamera();
    Methods::MeshRasterizer rasterizer;
    Methods::TrainableLight light;
    for (std::expected<void, Error> step :
         {rasterizer.open(device, std::size_t{size} * size, {0.0f, 0.0f, 0.0f}, true, 0.1f, 100.0f), light.open(device, lightSize)})
        if (Result opened = check(step); !opened)
            return opened;
    rasterizer.setLight(light.getLight());
    {
        CommandList list(device);
        light.recordBuild(list);
        if (Result submitted = submit(device, std::move(list)); !submitted)
            return submitted;
    }

    Core::Image target(size, size);
    target.setEncoding(Core::Image::Encoding::Srgb);
    Core::Image vertexColored(size, size);
    Core::Image textured(size, size);
    for (std::expected<void, Error> step :
         {rasterizer.upload(makeColoredSphere(true)), rasterizer.render(camera, target), rasterizer.upload(mesh), rasterizer.render(camera, vertexColored),
          rasterizer.setTexture(*atlas, texels), rasterizer.render(camera, textured)})
        if (Result result = check(step); !result)
            return result;
    const double bakeDifference = getMeanDifference(vertexColored, textured);
    std::println("{}x{} atlas, baked texture against vertex colours: mean difference {:.2e}", atlas->width, atlas->height, bakeDifference);
    if (bakeDifference > 2e-3)
        return std::unexpected(std::format("the baked texture differs from the vertex colours by {:.2e}", bakeDifference));

    std::vector<uint> rgba(std::size_t{size} * size);
    Core::toRgba8(target, rgba);
    Buffer<uint> targetBuffer;
    Buffer<float> first;
    Buffer<float> second;
    Training::Loss loss;
    Training::GpuAdam adam;
    const std::vector<float> zeros(texels.size(), 0.0f);
    for (std::expected<void, Error> step :
         {targetBuffer.open(device, rgba.size()), targetBuffer.upload(rgba), first.open(device, zeros.size()), second.open(device, zeros.size()),
          first.upload(zeros), second.upload(zeros), loss.open(device, size, size, Training::LossKind::L2), adam.open(device, {.epsilon = 1e-8f})})
        if (Result result = check(step); !result)
            return result;
    double startLoss = 0.0;
    double endLoss = 0.0;
    for (int step = 1; step <= fitSteps; ++step) {
        CommandList list(device);
        if (Result recorded = check(rasterizer.recordForward(list, camera)); !recorded)
            return recorded;
        loss.record(list, rasterizer.getImage(), targetBuffer.getPtr(), rasterizer.getImageGradient(), size, size);
        rasterizer.recordBackward(list);
        adam.record(list, {rasterizer.getTexels(), rasterizer.getTexelGradient(), first.getPtr(), second.getPtr(), rasterizer.getTexelValueCount(), 0.02f},
                    static_cast<std::uint32_t>(step));
        rasterizer.recordClamp(list);
        loss.recordReadback(list);
        if (Result submitted = submit(device, std::move(list)); !submitted)
            return submitted;
        std::expected<double, Error> value = loss.read();
        if (!value)
            return std::unexpected(value.error().message);
        if (step == 1)
            startLoss = *value;
        endLoss = *value;
    }
    std::println("fitting the texture to the inverted colours: loss {:.3e} -> {:.3e}", startLoss, endLoss);
    if (!(endLoss < startLoss * 0.01))
        return std::unexpected(std::format("the texture fit only lowers the loss from {:.3e} to {:.3e}", startLoss, endLoss));
    return {};
}

}
