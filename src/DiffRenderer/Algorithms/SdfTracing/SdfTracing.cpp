#include "DiffRenderer/Algorithms/SdfTracing/SdfTracing.h"

#include <cmath>

namespace DiffRenderer::Algorithms {

namespace {

constexpr float gridRate = 0.002f;
constexpr float bandWidth = 0.05f;
constexpr SdfSampling targetSampling{.samples = 256, .seed = 11};

const Description description{
    .id = "sdf-tracing",
    .name = "SDF sphere tracing",
    .representation = "a 32^3 grid SDF with cubic B-spline interpolation, seen from 6 views",
    .parameters = "all 32768 grid values",
    .discontinuities = "edge sampling along scanlines finds the silhouette crossings by bisection and adds their boundary term",
    .reference = "Vicini et al. differentiable-sdf-rendering (Mitsuba 3)",
    .hasBoundaryTerm = true,
};

}

const Description& SdfTracing::getDescription() const {
    return description;
}

std::expected<void, NoorRHI::Error> SdfTracing::open(NoorRHI::Device& device, const Settings& settings) {
    Core::Random random(settings.seed);
    scene.emplace(random);
    grid = scene->startGrid;
    gradient.assign(grid.size(), 0.0f);
    band.clear();
    for (std::uint32_t index = 0; index < scene->grid.size(); ++index)
        if (std::abs(scene->grid[index]) < bandWidth)
            band.push_back(index);
    boundaryTerm = settings.boundaryTerm;
    iteration = 0;
    if (std::expected<void, NoorRHI::Error> opened = tracer.open(device, *scene); !opened)
        return opened;
    target = Core::Image(SdfTracer::width, SdfTracer::height);
    current = Core::Image(SdfTracer::width, SdfTracer::height);
    imageGradient.assign(target.getPixels().size(), 0.0f);
    optimizer.reset();
    optimizer.setLearningRate(gridRate * settings.learningRateScale);
    return tracer.render(scene->grid, targetSampling, target.getPixels());
}

double SdfTracing::getBandError() const {
    double sum = 0.0;
    for (const std::uint32_t index : band)
        sum += std::abs(double{grid[index]} - scene->grid[index]);
    return sum / static_cast<double>(band.size());
}

std::expected<Step, NoorRHI::Error> SdfTracing::step() {
    const std::uint32_t seed = 2 * iteration + 9000;
    ++iteration;
    if (std::expected<void, NoorRHI::Error> rendered = tracer.render(grid, {.seed = seed}, current.getPixels()); !rendered)
        return std::unexpected(std::move(rendered.error()));
    const double loss = Core::getL2Loss(current, target, imageGradient);
    if (std::expected<void, NoorRHI::Error> derived = tracer.backward(imageGradient, {.seed = seed + 1, .boundary = boundaryTerm}, gradient); !derived)
        return std::unexpected(std::move(derived.error()));
    optimizer.step(grid, gradient);
    return Step{.loss = loss, .parameterError = getBandError()};
}

}
