#include "DiffRenderer/Training/Run.h"

#include <algorithm>
#include <chrono>

namespace DiffRenderer::Training {

namespace {

using Clock = std::chrono::steady_clock;

double getMilliseconds(const Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

}

std::expected<void, NoorRHI::Error> Run::open(Algorithms::Algorithm& opened, NoorRHI::Device& device, const Algorithms::Settings& chosen) {
    algorithm = nullptr;
    samples.clear();
    const Clock::time_point start = Clock::now();
    if (std::expected<void, NoorRHI::Error> result = opened.open(device, chosen); !result)
        return result;
    openMilliseconds = getMilliseconds(start);
    algorithm = &opened;
    settings = chosen;
    return {};
}

std::expected<void, NoorRHI::Error> Run::step() {
    const Clock::time_point start = Clock::now();
    std::expected<Algorithms::Step, NoorRHI::Error> taken = algorithm->step();
    if (!taken)
        return std::unexpected(std::move(taken.error()));
    const double milliseconds = getMilliseconds(start);
    const double error = Core::getMeanSquaredError(algorithm->getCurrent(), algorithm->getTarget());
    samples.push_back({.iteration = getIteration(), .loss = taken->loss, .psnr = Core::getPsnr(error), .parameterError = taken->parameterError,
                       .milliseconds = milliseconds});
    return {};
}

double Run::getMedianStepMilliseconds() const {
    if (samples.empty())
        return 0.0;
    std::vector<double> times;
    times.reserve(samples.size());
    for (const Sample& sample : samples)
        times.push_back(sample.milliseconds);
    std::ranges::nth_element(times, times.begin() + static_cast<std::ptrdiff_t>(times.size() / 2));
    return times[times.size() / 2];
}

}
