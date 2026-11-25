#include "DiffRenderer/Methods/Mesh/LargeStepOptimizer.h"

#include <array>
#include <cmath>
#include <utility>

#include "Smoothing.h"
#include "Smoothing.slang.h"

namespace DiffRenderer::Methods {

using namespace NoorRHI;

namespace {

constexpr std::uint32_t groupSize = 256;

}

std::expected<void, Error> LargeStepOptimizer::open(Device& target, const LargeStepSettings& chosen) {
    device = &target;
    settings = chosen;
    for (auto [pipeline, shader] : std::array<std::pair<ComputePipeline*, const Shader*>, 8>{{{&startSolve, &Smoothing::startSolve},
                                                                                              {&multiply, &Smoothing::multiply},
                                                                                              {&update, &Smoothing::update},
                                                                                              {&advance, &Smoothing::advance},
                                                                                              {&uniformMoments, &Smoothing::uniformMoments},
                                                                                              {&uniformStep, &Smoothing::uniformStep},
                                                                                              {&resetLargest, &Smoothing::resetLargest},
                                                                                              {&applyOffsets, &Smoothing::applyOffsets}}})
        if (std::expected<void, Error> opened = pipeline->open(target, *shader); !opened)
            return opened;
    return largestSecond.open(target, 1);
}

void LargeStepOptimizer::close() {
    *this = LargeStepOptimizer();
}

std::expected<void, Error> LargeStepOptimizer::setMesh(const SurfaceMesh& mesh) {
    std::vector<std::uint32_t> rowOffsets;
    std::vector<std::uint32_t> columns;
    mesh.findNeighbors(rowOffsets, columns);
    vertexCount = static_cast<std::uint32_t>(mesh.getVertexCount());
    step = 0;
    const std::size_t values = std::size_t{vertexCount} * 3;
    const std::vector<float> zeros(values, 0.0f);
    for (std::expected<void, Error> opened :
         {offsets.open(*device, rowOffsets.size()), neighbors.open(*device, columns.size()), solution.open(*device, values), residual.open(*device, values),
          direction.open(*device, values), product.open(*device, values), first.open(*device, values), second.open(*device, values),
          steps.open(*device, values), products.open(*device, std::size_t{getGroupCount()} * 3), residuals[0].open(*device, std::size_t{getGroupCount()} * 3),
          residuals[1].open(*device, std::size_t{getGroupCount()} * 3)})
        if (!opened)
            return opened;
    for (std::expected<void, Error> uploaded : {offsets.upload(rowOffsets), neighbors.upload(columns), first.upload(zeros), second.upload(zeros)})
        if (!uploaded)
            return uploaded;
    return {};
}

std::uint32_t LargeStepOptimizer::getGroupCount() const {
    return (vertexCount + groupSize - 1) / groupSize;
}

void LargeStepOptimizer::recordSolve(CommandList& list, const Ptr<float> right) const {
    Smoothing::SolveArguments arguments{offsets.getPtr(),      neighbors.getPtr(), right,
                                        solution.getPtr(),     residual.getPtr(),  direction.getPtr(),
                                        product.getPtr(),      products.getPtr(),  residuals[1].getPtr(),
                                        residuals[0].getPtr(), vertexCount,        getGroupCount(),
                                        settings.lambda};
    list.dispatch(startSolve, {vertexCount}, arguments);
    for (std::uint32_t iteration = 0; iteration < settings.solverIterations; ++iteration) {
        std::swap(arguments.residuals, arguments.nextResiduals);
        list.dispatch(multiply, {vertexCount}, arguments);
        list.dispatch(update, {vertexCount}, arguments);
        list.dispatch(advance, {vertexCount}, arguments);
    }
}

void LargeStepOptimizer::record(CommandList& list, const Ptr<float> positions, const Ptr<float> gradients, const float stepSize) {
    ++step;
    const std::uint32_t count = vertexCount * 3;
    const double firstCorrection = 1.0 - std::pow(double{settings.beta1}, step);
    const double secondCorrection = 1.0 - std::pow(double{settings.beta2}, step);
    recordSolve(list, gradients);
    const Smoothing::AdamArguments adam{solution.getPtr(),
                                        first.getPtr(),
                                        second.getPtr(),
                                        steps.getPtr(),
                                        largestSecond.getPtr(),
                                        count,
                                        settings.beta1,
                                        settings.beta2,
                                        static_cast<float>(stepSize / firstCorrection),
                                        static_cast<float>(1.0 / std::sqrt(secondCorrection)),
                                        settings.epsilon};
    list.dispatch(resetLargest, {1}, adam);
    list.dispatch(uniformMoments, {count}, adam);
    list.dispatch(uniformStep, {count}, adam);
    recordSolve(list, steps.getPtr());
    list.dispatch(applyOffsets, {count}, Smoothing::ApplyArguments{positions, solution.getPtr(), count});
}

}
