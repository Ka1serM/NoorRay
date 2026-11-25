#include "DiffRenderer/Training/Comparison.h"

#include <algorithm>
#include <chrono>
#include <format>
#include <random>
#include <system_error>

#include "DiffRenderer/Core/ImageFile.h"
#include "DiffRenderer/Core/Metrics.h"

namespace DiffRenderer::Training {

using namespace NoorRHI;

namespace {

using Clock = std::chrono::steady_clock;

double getMilliseconds(const Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

std::expected<void, Error> createFolder(const std::filesystem::path& folder) {
    std::error_code error;
    std::filesystem::create_directories(folder, error);
    if (error)
        return fail(ErrorCode::InvalidArgument, std::format("cannot create {}: {}", folder.string(), error.message()));
    return {};
}

std::expected<void, Error> save(const Core::Image& image, const std::filesystem::path& path) {
    if (std::expected<void, std::string> saved = Core::savePng(image, path); !saved)
        return fail(ErrorCode::InvalidArgument, saved.error());
    return {};
}

}

double MethodResult::getMedianStepMilliseconds() const {
    if (stepMilliseconds.empty())
        return 0.0;
    std::vector<float> times = stepMilliseconds;
    std::ranges::nth_element(times, times.begin() + static_cast<std::ptrdiff_t>(times.size() / 2));
    return times[times.size() / 2];
}

double MethodResult::getMeanRenderMilliseconds() const {
    if (views.empty())
        return 0.0;
    double sum = 0.0;
    for (const ViewResult& view : views)
        sum += view.renderMilliseconds;
    return sum / static_cast<double>(views.size());
}

std::filesystem::path Comparison::getRenderFolder(const Methods::Method& method) {
    return std::filesystem::path(method.getDescription().id);
}

std::filesystem::path Comparison::getImageName(const std::size_t testIndex) {
    return std::format("{:04}.png", testIndex);
}

std::expected<void, Error> Comparison::open(Device& target, const Scenes::Dataset& source, const std::span<Methods::Method* const> methods,
                                            const Budget& chosen, const Methods::Settings& chosenSettings, const std::filesystem::path& folder) {
    if (source.train.empty() || source.test.empty())
        return fail(ErrorCode::InvalidArgument, std::format("the dataset {} needs train and test views", source.id));
    if (chosen.iterations <= 0 || chosen.seconds < 0.0 || chosen.evaluateEvery < 0 || chosen.testStride == 0)
        return fail(ErrorCode::InvalidArgument, "the budget needs iterations above zero, no negative time or evaluation interval, and a test stride");
    for (const Methods::Method* method : methods)
        if (!method->supports(source.kind))
            return fail(ErrorCode::InvalidArgument, std::format("the method {} does not support the dataset {}", method->getDescription().id, source.id));
    device = &target;
    dataset = &source;
    budget = chosen;
    settings = chosenSettings;
    settings.iterations = chosen.iterations;
    output = folder;
    current = 0;
    methodOpen = false;
    orderEpoch = -1;
    results.assign(methods.size(), {});
    for (std::size_t index = 0; index < methods.size(); ++index)
        results[index].method = methods[index];
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    for (const Scenes::View& view : source.train) {
        width = std::max(width, view.image.getWidth());
        height = std::max(height, view.image.getHeight());
    }
    if (std::expected<void, Error> opened = loss.open(target, width, height, source.kind == Scenes::Kind::Volume ? LossKind::L2 : LossKind::L1Ssim); !opened)
        return opened;
    if (std::expected<void, Error> opened = targets.open(target, source); !opened)
        return opened;
    return writeReferences();
}

std::expected<void, Error> Comparison::writeReferences() const {
    if (output.empty())
        return {};
    const std::filesystem::path folder = output / getReferenceFolder();
    if (std::expected<void, Error> created = createFolder(folder); !created)
        return created;
    for (std::size_t index = 0; index < dataset->test.size(); ++index)
        if (std::expected<void, Error> saved = save(dataset->test[index].image, folder / getImageName(index)); !saved)
            return saved;
    return {};
}

std::expected<void, Error> Comparison::step() {
    if (isFinished())
        return {};
    if (!methodOpen)
        return openMethod();
    return train();
}

std::expected<void, Error> Comparison::runAll() {
    while (!isFinished())
        if (std::expected<void, Error> stepped = step(); !stepped)
            return stepped;
    return {};
}

std::expected<void, Error> Comparison::openMethod() {
    MethodResult& result = results[current];
    const Clock::time_point start = Clock::now();
    if (std::expected<void, Error> opened = result.method->open(*device, *dataset, settings); !opened)
        return fail(opened.error().code, std::format("{} does not open: {}", result.method->getDescription().id, opened.error().message));
    result.openMilliseconds = getMilliseconds(start);
    result.losses.reserve(static_cast<std::size_t>(budget.iterations));
    result.stepMilliseconds.reserve(static_cast<std::size_t>(budget.iterations));
    methodOpen = true;
    return {};
}

std::uint32_t Comparison::getTrainView(const int iteration) {
    const auto count = static_cast<int>(dataset->train.size());
    const int epoch = iteration / count;
    if (epoch != orderEpoch) {
        order.resize(dataset->train.size());
        for (std::uint32_t index = 0; index < order.size(); ++index)
            order[index] = index;
        std::mt19937_64 random(settings.seed * 0x9e3779b97f4a7c15ull + static_cast<std::uint64_t>(epoch));
        for (std::size_t index = order.size() - 1; index > 0; --index)
            std::swap(order[index], order[random() % (index + 1)]);
        orderEpoch = epoch;
    }
    return order[static_cast<std::size_t>(iteration % count)];
}

std::expected<void, Error> Comparison::train() {
    MethodResult& result = results[current];
    const int iteration = result.getIterations();
    const Clock::time_point start = Clock::now();
    std::expected<double, Error> stepped = result.method->step(targets.get(getTrainView(iteration)), loss);
    if (!stepped)
        return fail(stepped.error().code, std::format("{} failed at iteration {}: {}", result.method->getDescription().id, iteration, stepped.error().message));
    const double milliseconds = getMilliseconds(start);
    result.losses.push_back(static_cast<float>(*stepped));
    result.stepMilliseconds.push_back(static_cast<float>(milliseconds));
    result.trainSeconds += milliseconds / 1000.0;
    const bool last = result.getIterations() >= budget.iterations || (budget.seconds > 0.0 && result.trainSeconds >= budget.seconds);
    if (!last && budget.evaluateEvery > 0 && result.getIterations() % budget.evaluateEvery == 0)
        return evaluate(false);
    if (!last)
        return {};
    if (std::expected<void, Error> evaluated = evaluate(true); !evaluated)
        return evaluated;
    result.statistics = result.method->getStatistics();
    result.method->close();
    result.finished = true;
    methodOpen = false;
    ++current;
    orderEpoch = -1;
    return {};
}

std::expected<void, Error> Comparison::evaluate(const bool final) {
    MethodResult& result = results[current];
    const std::size_t stride = final ? 1 : budget.testStride;
    const std::filesystem::path folder = output / getRenderFolder(*result.method);
    if (final && !output.empty())
        if (std::expected<void, Error> created = createFolder(folder); !created)
            return created;
    Evaluation evaluation{result.getIterations(), result.trainSeconds, 0.0, 0.0};
    std::size_t count = 0;
    for (std::size_t index = 0; index < dataset->test.size(); index += stride) {
        const Scenes::View& view = dataset->test[index];
        if (render.getWidth() != view.camera.width || render.getHeight() != view.camera.height)
            render = Core::Image(view.camera.width, view.camera.height);
        const Clock::time_point start = Clock::now();
        if (std::expected<void, Error> rendered = result.method->render(view.camera, render); !rendered)
            return fail(rendered.error().code,
                        std::format("{} does not render {}: {}", result.method->getDescription().id, view.name, rendered.error().message));
        const double milliseconds = getMilliseconds(start);
        if (render.getWidth() != view.image.getWidth() || render.getHeight() != view.image.getHeight())
            return fail(ErrorCode::InvalidState,
                        std::format("{} renders {} at a different size than its image", result.method->getDescription().id, view.name));
        render.setEncoding(view.image.getEncoding());
        const double psnr = Core::getPsnr(Core::getMeanSquaredError(render, view.image));
        const double ssim = Core::getSsim(render, view.image);
        evaluation.psnr += psnr;
        evaluation.ssim += ssim;
        ++count;
        if (!final)
            continue;
        result.views.push_back({view.name, psnr, ssim, milliseconds});
        if (output.empty())
            continue;
        if (std::expected<void, Error> saved = save(render, folder / getImageName(index)); !saved)
            return saved;
    }
    evaluation.psnr /= static_cast<double>(count);
    evaluation.ssim /= static_cast<double>(count);
    result.evaluations.push_back(evaluation);
    return {};
}

}
