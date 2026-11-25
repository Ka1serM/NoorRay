#include "DiffRenderer/Algorithms/Rasterization/Rasterizer.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <utility>

#include "Rasterizing.slang.h"

namespace DiffRenderer::Algorithms {

using namespace NoorRHI;

std::expected<void, Error> Rasterizer::open(Device& target, const TexturedCube& cube, const bool antialiasing) {
    device = &target;
    const std::size_t pixels = std::size_t{TexturedCube::width} * TexturedCube::height;
    const std::size_t vertices = cube.getVertexCount();
    for (auto [pipeline, shader] : std::array<std::pair<ComputePipeline*, const Shader*>, 6>{{{&clear, &Rasterizing::clear},
                                                                                              {&rasterize, &Rasterizing::rasterize},
                                                                                              {&shade, &Rasterizing::shadePixels},
                                                                                              {&antialias, &Rasterizing::antialias},
                                                                                              {&antialiasBackward, &Rasterizing::antialiasBackward},
                                                                                              {&shadeBackward, &Rasterizing::shadeBackward}}})
        if (std::expected<void, Error> opened = pipeline->open(target, *shader); !opened)
            return opened;
    for (std::expected<void, Error> step :
         {positions.open(target, vertices), triangles.open(target, cube.triangles.size()), opposites.open(target, cube.opposites.size()),
          uvs.open(target, cube.uvs.size()), uvTriangles.open(target, cube.uvTriangles.size()), texture.open(target, cube.texture.size()),
          raster.open(target, pixels), color.open(target, pixels * 3), image.open(target, pixels * 3), imageGradient.open(target, pixels * 3),
          colorGradient.open(target, pixels * 3), positionGradient.open(target, vertices * 4), imageReadback.open(target, pixels * 3),
          positionReadback.open(target, vertices * 4)})
        if (!step)
            return step;
    for (std::expected<void, Error> step : {triangles.upload(cube.triangles), opposites.upload(cube.opposites), uvs.upload(cube.uvs),
                                            uvTriangles.upload(cube.uvTriangles), texture.upload(cube.texture)})
        if (!step)
            return step;
    arguments = {cube.background,
                 positions.getPtr(),
                 triangles.getPtr(),
                 opposites.getPtr(),
                 uvs.getPtr(),
                 uvTriangles.getPtr(),
                 texture.getPtr(),
                 raster.getPtr(),
                 color.getPtr(),
                 image.getPtr(),
                 imageGradient.getPtr(),
                 colorGradient.getPtr(),
                 positionGradient.getPtr(),
                 TexturedCube::width,
                 TexturedCube::height,
                 static_cast<uint>(cube.triangles.size() / 3),
                 static_cast<uint>(vertices),
                 TexturedCube::textureWidth,
                 TexturedCube::textureHeight,
                 antialiasing ? 1u : 0u};
    clearCount = static_cast<uint>(std::max(pixels * 3, vertices * 4));
    return {};
}

void Rasterizer::record(CommandList& list, const std::span<const ComputePipeline* const> pipelines) const {
    for (const ComputePipeline* pipeline : pipelines)
        list.dispatch(*pipeline, pipeline == &clear ? DispatchSize{clearCount} : DispatchSize{arguments.width, arguments.height}, arguments);
}

std::expected<void, Error> Rasterizer::submit(CommandList&& list) {
    std::expected<Token, Error> token = device->submit(std::move(list));
    if (!token)
        return std::unexpected(std::move(token.error()));
    return device->wait(*token);
}

std::expected<void, Error> Rasterizer::render(const std::span<const float4> clip, const std::span<float> result) {
    CommandList list(*device);
    list.upload(positions, clip);
    const std::array<const ComputePipeline*, 4> passes{&clear, &rasterize, &shade, &antialias};
    record(list, passes);
    list.download(image, imageReadback);
    if (std::expected<void, Error> submitted = submit(std::move(list)); !submitted)
        return submitted;
    std::expected<std::span<const Atomic<float>>, Error> values = imageReadback.read();
    if (!values)
        return std::unexpected(std::move(values.error()));
    std::memcpy(result.data(), values->data(), result.size_bytes());
    return {};
}

std::expected<void, Error> Rasterizer::backward(const std::span<const float> gradient, const std::span<float4> clipGradient) {
    CommandList list(*device);
    list.upload(imageGradient, gradient);
    const std::array<const ComputePipeline*, 2> passes{&antialiasBackward, &shadeBackward};
    record(list, passes);
    list.download(positionGradient, positionReadback);
    if (std::expected<void, Error> submitted = submit(std::move(list)); !submitted)
        return submitted;
    std::expected<std::span<const Atomic<float>>, Error> values = positionReadback.read();
    if (!values)
        return std::unexpected(std::move(values.error()));
    std::memcpy(clipGradient.data(), values->data(), clipGradient.size_bytes());
    return {};
}

}
