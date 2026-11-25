#include "DiffRenderer/Algorithms/Rasterization/Rasterization.h"

#include <cmath>

namespace DiffRenderer::Algorithms {

namespace {

constexpr float learningRate = 0.01f;

const Description description{
    .id = "rasterization",
    .name = "Triangle rasterization",
    .representation = "a textured cube: 8 vertices, 12 triangles, a 64x64 texture",
    .parameters = "vertex positions",
    .discontinuities = "analytic antialiasing as in nvdiffrast: where a silhouette edge passes between two pixel centres, the two pixels are blended by "
                       "the crossing point of the edge, which gives gradients for the edge vertices",
    .reference = "nvdiffrast",
    .hasBoundaryTerm = true,
};

double getMeanError(const std::vector<float>& values, const std::vector<float>& expected) {
    double sum = 0.0;
    for (std::size_t index = 0; index < values.size(); ++index)
        sum += std::abs(double{values[index]} - expected[index]);
    return sum / static_cast<double>(values.size());
}

}

const Description& Rasterization::getDescription() const {
    return description;
}

std::expected<void, NoorRHI::Error> Rasterization::open(NoorRHI::Device& device, const Settings& settings) {
    Core::Random random(settings.seed);
    positions = cube.perturb(random);
    gradients.assign(positions.size(), 0.0f);
    clip.resize(cube.getVertexCount());
    clipGradient.resize(cube.getVertexCount());
    if (std::expected<void, NoorRHI::Error> opened = rasterizer.open(device, cube, settings.boundaryTerm); !opened)
        return opened;
    target = Core::Image(TexturedCube::width, TexturedCube::height);
    current = Core::Image(TexturedCube::width, TexturedCube::height);
    imageGradient.assign(target.getPixels().size(), 0.0f);
    optimizer.reset();
    optimizer.setLearningRate(learningRate * settings.learningRateScale);
    cube.project(cube.positions, clip);
    return rasterizer.render(clip, target.getPixels());
}

std::expected<Step, NoorRHI::Error> Rasterization::step() {
    cube.project(positions, clip);
    if (std::expected<void, NoorRHI::Error> rendered = rasterizer.render(clip, current.getPixels()); !rendered)
        return std::unexpected(std::move(rendered.error()));
    const double loss = Core::getL2Loss(current, target, imageGradient);
    if (std::expected<void, NoorRHI::Error> derived = rasterizer.backward(imageGradient, clipGradient); !derived)
        return std::unexpected(std::move(derived.error()));
    cube.pullBack(clipGradient, gradients);
    optimizer.step(positions, gradients);
    return Step{.loss = loss, .parameterError = getMeanError(positions, cube.positions)};
}

}
