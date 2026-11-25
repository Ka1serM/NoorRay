#include "DiffRenderer/Methods/Mesh/MeshRasterizer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <numbers>
#include <utility>
#include <vector>

#include "MeshRasterizing.slang.h"

namespace DiffRenderer::Methods {

using namespace NoorRHI;

namespace {

std::span<const float> asFloats(const std::span<const Vector3> vectors) {
    return {reinterpret_cast<const float*>(vectors.data()), vectors.size() * 3};
}

std::span<float> asWritableFloats(const std::span<Vector3> vectors) {
    return {reinterpret_cast<float*>(vectors.data()), vectors.size() * 3};
}

constexpr std::uint32_t splitSumSamples = 1024;

//! The radical inverse in base 2, the second coordinate of the Hammersley set.
double getRadicalInverse(std::uint32_t bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return static_cast<double>(bits) / 4294967296.0;
}

//! The scale and the bias of F0 in the split sum of the GGX BRDF with the height correlated Smith term and alpha = roughness^2, as bsdf_256_256.bin of nvdiffrec.
float2 integrateSplitSum(const double cosine, const double roughness) {
    const double alpha = roughness * roughness;
    const double alphaSquared = alpha * alpha;
    const double sine = std::sqrt(1.0 - cosine * cosine);
    double scale = 0.0;
    double bias = 0.0;
    for (std::uint32_t sample = 0; sample < splitSumSamples; ++sample) {
        const double u = static_cast<double>(sample) / splitSumSamples;
        const double v = getRadicalInverse(sample);
        const double phi = 2.0 * std::numbers::pi * u;
        const double hz = std::sqrt((1.0 - v) / (1.0 + (alphaSquared - 1.0) * v));
        const double hr = std::sqrt(1.0 - hz * hz);
        const double hx = hr * std::cos(phi);
        const double viewHalf = sine * hx + cosine * hz;
        const double lightZ = 2.0 * viewHalf * hz - cosine;
        if (lightZ <= 0.0)
            continue;
        const double visibleView = lightZ * std::sqrt(cosine * cosine * (1.0 - alphaSquared) + alphaSquared);
        const double visibleLight = cosine * std::sqrt(lightZ * lightZ * (1.0 - alphaSquared) + alphaSquared);
        const double shadowing = 2.0 * cosine * lightZ / (visibleView + visibleLight);
        const double weight = shadowing * std::max(viewHalf, 0.0) / (hz * cosine);
        const double fresnel = std::pow(1.0 - std::max(viewHalf, 0.0), 5.0);
        scale += (1.0 - fresnel) * weight;
        bias += fresnel * weight;
    }
    return {static_cast<float>(scale / splitSumSamples), static_cast<float>(bias / splitSumSamples)};
}

std::vector<float2> makeSplitSum() {
    constexpr std::uint32_t size = MeshRasterizing::splitSumSize;
    std::vector<float2> table(std::size_t{size} * size);
    for (std::uint32_t row = 0; row < size; ++row)
        for (std::uint32_t column = 0; column < size; ++column)
            table[std::size_t{row} * size + column] = integrateSplitSum((column + 0.5) / size, (row + 0.5) / size);
    return table;
}

std::vector<float> interleave(const SurfaceMesh& mesh) {
    std::vector<float> values(mesh.getVertexCount() * MeshRasterizing::vertexChannels);
    for (std::size_t vertex = 0; vertex < mesh.getVertexCount(); ++vertex) {
        const Vector3 kd = mesh.colors[vertex];
        const Vector3 ks = mesh.materials[vertex];
        std::ranges::copy(std::array{kd.x, kd.y, kd.z, ks.x, ks.y, ks.z}, values.begin() + static_cast<std::ptrdiff_t>(vertex * MeshRasterizing::vertexChannels));
    }
    return values;
}

std::expected<void, Error> submit(Device& device, CommandList&& list) {
    std::expected<Token, Error> token = device.submit(std::move(list));
    if (!token)
        return std::unexpected(std::move(token.error()));
    return device.wait(*token);
}

}

std::expected<void, Error> MeshRasterizer::open(Device& target, const std::size_t maxPixels, const float3 background, const bool antialiasing,
                                                const float nearDistance, const float farDistance) {
    device = &target;
    nearPlane = nearDistance;
    farPlane = farDistance;
    for (auto [pipeline, shader] : std::array<std::pair<ComputePipeline*, const Shader*>, 14>{{{&clear, &MeshRasterizing::clear},
                                                                                              {&project, &MeshRasterizing::projectVertices},
                                                                                              {&accumulateNormals, &MeshRasterizing::accumulateNormals},
                                                                                              {&finishNormals, &MeshRasterizing::finishNormals},
                                                                                              {&normalBackward, &MeshRasterizing::normalBackward},
                                                                                              {&faceNormalBackward, &MeshRasterizing::faceNormalBackward},
                                                                                              {&clampMaterials, &MeshRasterizing::clampMaterials},
                                                                                              {&resolve, &MeshRasterizing::resolve},
                                                                                              {&shade, &MeshRasterizing::shadePixels},
                                                                                              {&antialias, &MeshRasterizing::antialias},
                                                                                              {&antialiasBackward, &MeshRasterizing::antialiasBackward},
                                                                                              {&shadeBackward, &MeshRasterizing::shadeBackward},
                                                                                              {&gatherGradients, &MeshRasterizing::gatherGradients},
                                                                                              {&gatherTexelGradients, &MeshRasterizing::gatherTexelGradients}}})
        if (std::expected<void, Error> opened = pipeline->open(target, *shader); !opened)
            return opened;
    const GraphicsPipelineDesc rasterDesc{.vertex = MeshRasterizing::rasterVertex,
                                          .fragment = MeshRasterizing::rasterFragment,
                                          .state = {.cull = CullMode::None, .depthCompare = CompareOp::Less, .depthTest = true, .depthWrite = true},
                                          .colorFormat = ImageFormat::R32Float,
                                          .depthFormat = ImageFormat::D32Float};
    if (std::expected<void, Error> opened = raster.open(target, rasterDesc); !opened)
        return opened;
    for (std::expected<void, Error> step : {rasterBuffer.open(target, maxPixels), shaded.open(target, maxPixels * 3), image.open(target, maxPixels * 3),
                                            imageGradient.open(target, maxPixels * 3), shadedGradient.open(target, maxPixels * 3),
                                            splitSum.open(target, std::size_t{MeshRasterizing::splitSumSize} * MeshRasterizing::splitSumSize)})
        if (!step)
            return step;
    if (std::expected<void, Error> uploaded = splitSum.upload(makeSplitSum()); !uploaded)
        return uploaded;
    arguments = {};
    arguments.background = {background.x, background.y, background.z, 0.0f};
    arguments.raster = rasterBuffer.getPtr();
    arguments.shaded = shaded.getPtr();
    arguments.image = image.getPtr();
    arguments.imageGradient = imageGradient.getPtr();
    arguments.shadedGradient = shadedGradient.getPtr();
    arguments.splitSum = splitSum.getPtr();
    arguments.antialiasing = antialiasing ? 1u : 0u;
    arguments.minimumRoughness = EnvironmentLight::minimumRoughness;
    return {};
}

void MeshRasterizer::setRegularizers(const std::uint32_t seed, const float smoothWeight, const float occlusionWeight) {
    arguments.seed = seed;
    arguments.smoothWeight = smoothWeight;
    arguments.occlusionWeight = occlusionWeight;
}

void MeshRasterizer::close() {
    *this = MeshRasterizer();
}

std::expected<void, Error> MeshRasterizer::upload(const SurfaceMesh& mesh) {
    const std::size_t vertices = mesh.getVertexCount();
    const std::vector<std::int32_t> oppositeVertices = mesh.findOppositeVertices();
    const std::size_t materialValues = vertices * MeshRasterizing::vertexChannels;
    for (std::expected<void, Error> step :
         {positions.open(*device, vertices * 3), vertexMaterials.open(*device, materialValues), triangles.open(*device, mesh.triangles.size()),
          opposites.open(*device, oppositeVertices.size()), clip.open(*device, vertices), clipGradient.open(*device, vertices * 4),
          vertexMaterialAccumulator.open(*device, materialValues), positionGradient.open(*device, vertices * 3),
          vertexMaterialGradient.open(*device, materialValues), normalSums.open(*device, vertices * 3), tangentSums.open(*device, vertices * 3),
          normals.open(*device, vertices * 3), tangents.open(*device, vertices * 3), normalGradient.open(*device, vertices * 3),
          normalSumGradient.open(*device, vertices * 3), shadingPositionGradient.open(*device, vertices * 3)})
        if (!step)
            return step;
    for (std::expected<void, Error> step : {positions.upload(asFloats(mesh.positions)), vertexMaterials.upload(interleave(mesh)),
                                            triangles.upload(mesh.triangles), opposites.upload(oppositeVertices)})
        if (!step)
            return step;
    arguments.positions = positions.getPtr();
    arguments.vertexMaterials = vertexMaterials.getPtr();
    arguments.normalSums = normalSums.getPtr();
    arguments.tangentSums = tangentSums.getPtr();
    arguments.normals = normals.getPtr();
    arguments.tangents = tangents.getPtr();
    arguments.normalGradient = normalGradient.getPtr();
    arguments.normalSumGradient = normalSumGradient.getPtr();
    arguments.shadingPositionGradient = shadingPositionGradient.getPtr();
    arguments.triangles = triangles.getPtr();
    arguments.opposites = opposites.getPtr();
    arguments.clip = clip.getPtr();
    arguments.clipGradient = clipGradient.getPtr();
    arguments.vertexMaterialAccumulator = vertexMaterialAccumulator.getPtr();
    arguments.positionGradient = positionGradient.getPtr();
    arguments.vertexMaterialGradient = vertexMaterialGradient.getPtr();
    arguments.vertexCount = static_cast<uint>(vertices);
    arguments.triangleCount = static_cast<uint>(mesh.getTriangleCount());
    arguments.textureWidth = 0;
    arguments.textureHeight = 0;
    return {};
}

std::expected<void, Error> MeshRasterizer::setTexture(const TextureAtlas& atlas, const std::span<const float> values) {
    for (std::expected<void, Error> step :
         {uvs.open(*device, atlas.uvs.size()), uvTriangles.open(*device, atlas.uvTriangles.size()), texels.open(*device, values.size()),
          texelAccumulator.open(*device, values.size()), texelGradient.open(*device, values.size()), uvs.upload(atlas.uvs),
          uvTriangles.upload(atlas.uvTriangles), texels.upload(values)})
        if (!step)
            return step;
    arguments.uvs = uvs.getPtr();
    arguments.uvTriangles = uvTriangles.getPtr();
    arguments.texels = texels.getPtr();
    arguments.texelAccumulator = texelAccumulator.getPtr();
    arguments.texelGradient = texelGradient.getPtr();
    arguments.textureWidth = atlas.width;
    arguments.textureHeight = atlas.height;
    return {};
}

std::expected<void, Error> MeshRasterizer::downloadTexture(const std::span<float> values) const {
    return texels.download(values);
}

std::expected<void, Error> MeshRasterizer::download(SurfaceMesh& mesh) const {
    mesh.positions.resize(arguments.vertexCount);
    mesh.colors.resize(arguments.vertexCount);
    mesh.materials.resize(arguments.vertexCount);
    std::vector<float> values(vertexMaterials.getSize());
    for (std::expected<void, Error> step : {positions.download(asWritableFloats(mesh.positions)), vertexMaterials.download(values)})
        if (!step)
            return step;
    for (std::size_t vertex = 0; vertex < arguments.vertexCount; ++vertex) {
        const float* value = values.data() + vertex * MeshRasterizing::vertexChannels;
        mesh.colors[vertex] = {value[0], value[1], value[2]};
        mesh.materials[vertex] = {value[3], value[4], value[5]};
    }
    return {};
}

std::expected<void, Error> MeshRasterizer::openTargets(const std::uint32_t width, const std::uint32_t height) {
    if (ids && ids.getWidth() == width && ids.getHeight() == height)
        return {};
    if (std::size_t{width} * height > rasterBuffer.getSize())
        return fail(ErrorCode::InvalidArgument, "the view has more pixels than the rasterizer was opened for");
    if (std::expected<void, Error> opened =
            ids.open(*device, {.width = width, .height = height, .format = ImageFormat::R32Float, .usage = ImageUsage::ColorAttachment | ImageUsage::Sampled});
        !opened)
        return opened;
    return depth.open(*device, {.width = width, .height = height, .format = ImageFormat::D32Float, .usage = ImageUsage::DepthAttachment});
}

MeshRasterizing::Projection MeshRasterizer::makeProjection(const Scenes::Camera& camera) const {
    const auto width = static_cast<float>(camera.width);
    const auto height = static_cast<float>(camera.height);
    const float depthScale = farPlane / (farPlane - nearPlane);
    return {camera.worldToCamera[0],
            camera.worldToCamera[1],
            camera.worldToCamera[2],
            {2.0f * camera.intrinsics.x / width, 2.0f * camera.intrinsics.y / height, 2.0f * camera.intrinsics.z / width - 1.0f,
             2.0f * camera.intrinsics.w / height - 1.0f},
            {depthScale, -depthScale * nearPlane},
            camera.width,
            camera.height};
}

std::expected<void, Error> MeshRasterizer::recordForward(CommandList& list, const Scenes::Camera& camera) {
    if (std::expected<void, Error> opened = openTargets(camera.width, camera.height); !opened)
        return opened;
    arguments.projection = makeProjection(camera);
    arguments.ids = ids.getSampledHandle();
    const float3 position = camera.getPosition();
    arguments.cameraPosition = {position.x, position.y, position.z, 0.0f};
    const DispatchSize pixels{camera.width, camera.height};
    list.dispatch(clear, {std::max(arguments.vertexCount * MeshRasterizing::vertexChannels, getTexelValueCount())}, arguments);
    list.dispatch(project, {arguments.vertexCount}, arguments);
    list.dispatch(accumulateNormals, {arguments.triangleCount}, arguments);
    list.dispatch(finishNormals, {arguments.vertexCount}, arguments);
    list.beginRender({.color = &ids, .depth = &depth, .clearColor = {0.0f, 0.0f, 0.0f, 0.0f}});
    list.draw(raster, arguments.triangleCount * 3, 1, arguments);
    list.endRender();
    list.dispatch(resolve, pixels, arguments);
    list.dispatch(shade, pixels, arguments);
    list.dispatch(antialias, pixels, arguments);
    return {};
}

void MeshRasterizer::recordBackward(CommandList& list) const {
    const DispatchSize pixels{arguments.projection.width, arguments.projection.height};
    list.dispatch(antialiasBackward, pixels, arguments);
    list.dispatch(shadeBackward, pixels, arguments);
    list.dispatch(normalBackward, {arguments.vertexCount}, arguments);
    list.dispatch(faceNormalBackward, {arguments.triangleCount}, arguments);
    list.dispatch(gatherGradients, {arguments.vertexCount * MeshRasterizing::vertexChannels}, arguments);
    if (arguments.textureWidth > 0)
        list.dispatch(gatherTexelGradients, {getTexelValueCount()}, arguments);
}

void MeshRasterizer::recordClamp(CommandList& list) const {
    list.dispatch(clampMaterials, {arguments.textureWidth > 0 ? arguments.textureWidth * arguments.textureHeight : arguments.vertexCount}, arguments);
}

std::expected<void, Error> MeshRasterizer::render(const Scenes::Camera& camera, Core::Image& result) {
    const std::size_t values = std::size_t{camera.width} * camera.height * 3;
    if (imageReadback.getSize() != values)
        if (std::expected<void, Error> opened = imageReadback.open(*device, values); !opened)
            return opened;
    CommandList list(*device);
    if (std::expected<void, Error> recorded = recordForward(list, camera); !recorded)
        return recorded;
    list.download(image, imageReadback);
    if (std::expected<void, Error> submitted = submit(*device, std::move(list)); !submitted)
        return submitted;
    std::expected<std::span<const float>, Error> read = imageReadback.read();
    if (!read)
        return std::unexpected(std::move(read.error()));
    std::ranges::copy(*read, result.getPixels().begin());
    return {};
}

}
