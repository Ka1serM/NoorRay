#include <algorithm>
#include <array>
#include <filesystem>
#include <format>
#include <print>
#include <random>
#include <vector>

#include "DiffRenderer.Tests/Tests.h"
#include "DiffRenderer/Training/Comparison.h"
#include "DiffRenderer/Training/ComparisonReport.h"

namespace DiffRenderer::Tests {

using namespace NoorRHI;

namespace {

constexpr std::uint32_t width = 40;
constexpr std::uint32_t height = 30;
constexpr std::size_t valueCount = std::size_t{width} * height * 3;
constexpr std::array<float, 3> trueColor{0.2f, 0.5f, 0.7f};

//! Learns one colour for the whole image, with plain gradient descent on the shared loss.
class ConstantColor final : public Methods::Method {
public:
    explicit ConstantColor(const std::string_view id) : description{.id = id, .name = id, .scene = true, .object = true} {}

    const Methods::Description& getDescription() const override { return description; }

    std::expected<void, Error> open(Device& opened, const Scenes::Dataset&, const Methods::Settings&) override {
        device = &opened;
        color = {0.5f, 0.5f, 0.5f};
        views.clear();
        for (std::expected<void, Error> step : {image.open(opened, valueCount), gradient.open(opened, valueCount), readback.open(opened, valueCount)})
            if (!step)
                return step;
        return {};
    }

    std::expected<double, Error> step(const Methods::Target& target, Training::Loss& loss) override {
        views.push_back(target.index);
        std::vector<float> values(valueCount);
        fill(values);
        CommandList list(*device);
        list.upload(image, std::span<const float>(values));
        loss.record(list, image.getPtr(), target.image, gradient.getPtr(), width, height);
        loss.recordReadback(list);
        list.download(gradient, readback);
        std::expected<Token, Error> token = device->submit(std::move(list));
        if (!token)
            return std::unexpected(std::move(token.error()));
        if (std::expected<void, Error> waited = device->wait(*token); !waited)
            return std::unexpected(std::move(waited.error()));
        std::expected<std::span<const float>, Error> gradients = readback.read();
        if (!gradients)
            return std::unexpected(std::move(gradients.error()));
        std::array<float, 3> sums{};
        for (std::size_t index = 0; index < gradients->size(); ++index)
            sums[index % 3] += (*gradients)[index];
        for (std::size_t channel = 0; channel < 3; ++channel)
            color[channel] -= 0.1f * sums[channel];
        return loss.read();
    }

    std::expected<void, Error> render(const Scenes::Camera& camera, Core::Image& result) override {
        result = Core::Image(camera.width, camera.height);
        std::vector<float> values(result.getPixels().size());
        fill(values);
        std::ranges::copy(values, result.getPixels().begin());
        return {};
    }

    Methods::Statistics getStatistics() const override { return {1, sizeof(color)}; }

    std::vector<std::uint32_t> views;
    std::array<float, 3> color{};

private:
    void fill(std::vector<float>& values) const {
        for (std::size_t index = 0; index < values.size(); ++index)
            values[index] = color[index % 3];
    }

    Methods::Description description;
    Device* device = nullptr;
    Buffer<float> image;
    Buffer<float> gradient;
    Readback<float> readback;
};

Scenes::View makeView(std::mt19937& random, const std::size_t index) {
    std::normal_distribution<float> noise(0.0f, 0.05f);
    Scenes::View view;
    view.name = std::format("view{}", index);
    view.camera.width = width;
    view.camera.height = height;
    view.image = Core::Image(width, height);
    view.image.setEncoding(Core::Image::Encoding::Srgb);
    for (std::size_t value = 0; value < valueCount; ++value)
        view.image.getPixels()[value] = std::clamp(trueColor[value % 3] + noise(random), 0.0f, 1.0f);
    return view;
}

}

Result testComparison(Device& device) {
    std::mt19937 random(13);
    Scenes::Dataset dataset;
    dataset.id = "constant";
    dataset.name = "Constant colour";
    dataset.kind = Scenes::Kind::Object;
    for (std::size_t index = 0; index < 7; ++index)
        dataset.train.push_back(makeView(random, index));
    for (std::size_t index = 0; index < 5; ++index)
        dataset.test.push_back(makeView(random, index));
    ConstantColor first("constant-a");
    ConstantColor second("constant-b");
    const std::array<Methods::Method*, 2> methods{&first, &second};
    const std::filesystem::path output = "/var/tmp/diffrenderer-comparison-test";
    Training::Comparison comparison;
    const Training::Budget budget{.iterations = 60, .seconds = 0.0, .evaluateEvery = 20, .testStride = 2};
    if (Result opened = check(comparison.open(device, dataset, methods, budget, {}, output)); !opened)
        return opened;
    int steps = 0;
    while (!comparison.isFinished()) {
        if (Result stepped = check(comparison.step()); !stepped)
            return stepped;
        ++steps;
    }
    if (Result written = check(Training::writeComparisonReport(comparison)); !written)
        return written;
    std::print("{}", Training::getComparisonTable(comparison));
    std::println("{} step calls; colour of the first method ({:.3f}, {:.3f}, {:.3f})", steps, first.color[0], first.color[1], first.color[2]);
    if (first.views != second.views)
        return std::unexpected("the methods saw different view orders");
    for (std::size_t epoch = 0; epoch + 1 < first.views.size() / dataset.train.size(); ++epoch) {
        std::vector<std::uint32_t> views(first.views.begin() + static_cast<std::ptrdiff_t>(epoch * 7),
                                         first.views.begin() + static_cast<std::ptrdiff_t>(epoch * 7 + 7));
        std::ranges::sort(views);
        if (views != std::vector<std::uint32_t>{0, 1, 2, 3, 4, 5, 6})
            return std::unexpected(std::format("epoch {} is not a permutation of the train views", epoch));
    }
    for (const Training::MethodResult& result : comparison.getResults()) {
        if (result.getIterations() != 60 || result.evaluations.size() != 3 || result.views.size() != 5)
            return std::unexpected("a method has the wrong number of iterations, evaluations or views");
        if (result.evaluations.back().psnr < 24.0 || result.losses.back() > 0.5f * result.losses.front())
            return std::unexpected("the constant colour did not converge");
    }
    for (const std::filesystem::path& file :
         {output / "report.json", output / "report.md", output / "references" / "0004.png", output / "constant-b" / "0004.png"})
        if (!std::filesystem::exists(file))
            return std::unexpected(std::format("{} is missing", file.string()));
    std::println("view order identical for both methods and a permutation in each epoch; report in {}", output.string());
    return {};
}

}
