#include "DiffRenderer/Algorithms/GaussianSplatting/Splatter.h"

#include <algorithm>
#include <utility>

#include "Splatting.slang.h"

namespace DiffRenderer::Algorithms {

using namespace NoorRHI;

namespace {

constexpr std::size_t projectedGradientStride = 9;

Splatting::Parameters getPointers(const std::array<Buffer<float>, 5>& buffers) {
    return {buffers[0].getPtr(), buffers[1].getPtr(), buffers[2].getPtr(), buffers[3].getPtr(), buffers[4].getPtr()};
}

}

std::expected<void, Error> Splatter::open(Device& target, const std::size_t count, const SplatCamera& camera) {
    device = &target;
    const std::size_t pixels = std::size_t{camera.width} * camera.height;
    for (auto [pipeline, shader] : std::array<std::pair<ComputePipeline*, const Shader*>, 5>{{{&project, &Splatting::projectGaussians},
                                                                                               {&sort, &Splatting::sortGaussians},
                                                                                               {&composite, &Splatting::composite},
                                                                                               {&compositeBackward, &Splatting::compositeBackward},
                                                                                               {&projectBackward, &Splatting::projectBackward}}})
        if (std::expected<void, Error> opened = pipeline->open(target, *shader); !opened)
            return opened;
    for (std::size_t group = 0; group < parameters.size(); ++group)
        for (std::expected<void, Error> step : {parameters[group].open(target, count * Gaussians::groupSizes[group]),
                                                gradients[group].open(target, count * Gaussians::groupSizes[group]),
                                                gradientReadbacks[group].open(target, count * Gaussians::groupSizes[group])})
            if (!step)
                return step;
    for (std::expected<void, Error> step :
         {projected.open(target, count * 2), order.open(target, count), image.open(target, pixels * 3), transmittance.open(target, pixels),
          contributors.open(target, pixels), imageGradient.open(target, pixels * 3), projectedGradient.open(target, count * projectedGradientStride),
          imageReadback.open(target, pixels * 3)})
        if (!step)
            return step;
    arguments = {{camera.view[0], camera.view[1], camera.view[2]},
                 camera.intrinsics,
                 camera.background,
                 getPointers(parameters),
                 getPointers(gradients),
                 projected.getPtr(),
                 order.getPtr(),
                 image.getPtr(),
                 transmittance.getPtr(),
                 contributors.getPtr(),
                 imageGradient.getPtr(),
                 projectedGradient.getPtr(),
                 static_cast<uint>(count),
                 camera.width,
                 camera.height,
                 camera.nearPlane,
                 camera.farPlane,
                 camera.blur};
    return {};
}

void Splatter::record(CommandList& list, const std::span<const Stage> stages) const {
    for (const Stage& stage : stages)
        list.dispatch(*stage.pipeline, stage.size, arguments);
}

std::expected<void, Error> Splatter::submit(CommandList&& list) {
    std::expected<Token, Error> token = device->submit(std::move(list));
    if (!token)
        return std::unexpected(std::move(token.error()));
    return device->wait(*token);
}

std::expected<void, Error> Splatter::render(const Gaussians& gaussians, const std::span<float> result) {
    CommandList list(*device);
    for (std::size_t group = 0; group < parameters.size(); ++group)
        list.upload(parameters[group], std::span<const float>(*gaussians.getGroups()[group]));
    const DispatchSize perGaussian{arguments.count};
    const std::array<Stage, 3> stages{{{&project, perGaussian}, {&sort, perGaussian}, {&composite, {arguments.width, arguments.height}}}};
    record(list, stages);
    list.download(image, imageReadback);
    if (std::expected<void, Error> submitted = submit(std::move(list)); !submitted)
        return submitted;
    std::expected<std::span<const float>, Error> values = imageReadback.read();
    if (!values)
        return std::unexpected(std::move(values.error()));
    std::ranges::copy(*values, result.begin());
    return {};
}

std::expected<void, Error> Splatter::backward(const std::span<const float> gradient, Gaussians& result) {
    CommandList list(*device);
    list.upload(imageGradient, gradient);
    const std::array<Stage, 2> stages{{{&compositeBackward, {arguments.width, arguments.height}}, {&projectBackward, {arguments.count}}}};
    record(list, stages);
    for (std::size_t group = 0; group < gradients.size(); ++group)
        list.download(gradients[group], gradientReadbacks[group]);
    if (std::expected<void, Error> submitted = submit(std::move(list)); !submitted)
        return submitted;
    for (std::size_t group = 0; group < gradients.size(); ++group) {
        std::expected<std::span<const float>, Error> values = gradientReadbacks[group].read();
        if (!values)
            return std::unexpected(std::move(values.error()));
        std::ranges::copy(*values, result.getGroups()[group]->begin());
    }
    return {};
}

}
