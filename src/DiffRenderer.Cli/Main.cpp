#include <algorithm>
#include <array>
#include <charconv>
#include <concepts>
#include <expected>
#include <filesystem>
#include <optional>
#include <print>
#include <string_view>
#include <vector>

#include "DiffRenderer/Algorithms/Catalog.h"
#include "DiffRenderer/Core/ImageFile.h"
#include "DiffRenderer/Methods/Catalog.h"
#include "DiffRenderer/Scenes/Dataset.h"
#include "DiffRenderer/Training/Comparison.h"
#include "DiffRenderer/Training/ComparisonReport.h"
#include "DiffRenderer/Training/Report.h"

namespace {

using namespace DiffRenderer;

struct Options {
    std::vector<std::string_view> algorithms;
    int iterations = 300;
    Algorithms::Settings settings;
    std::optional<std::filesystem::path> report;
    std::optional<std::string_view> dataset;
    Scenes::LoadOptions loadOptions;
    std::optional<std::string_view> scenario;
    std::vector<std::string_view> methods;
    Training::Budget budget;
    std::filesystem::path output;
    bool list = false;
};

constexpr std::string_view usage =
    "usage: diffrenderer [--list] [--algorithm <id>|all]... [--iterations <n>] [--seed <n>] [--learning-rate-scale <x>]\n"
    "                    [--no-boundary-term] [--report <file.json>]\n"
    "       diffrenderer --dataset <id> [--downscale <n>]\n"
    "       diffrenderer --scenario <dataset id> [--method <id>|all]... [--iterations <n>] [--seconds <s>] [--evaluate-every <n>]\n"
    "                    [--test-stride <n>] [--downscale <n>] [--seed <n>] [--out <folder>]";

template <class T>
std::optional<T> parse(const std::string_view text) {
    T value{};
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc() || end != text.data() + text.size())
        return std::nullopt;
    return value;
}

std::expected<Options, std::string> parseOptions(const int count, char** arguments) {
    Options options;
    for (int index = 1; index < count; ++index) {
        const std::string_view name = arguments[index];
        if (name == "--list") {
            options.list = true;
            continue;
        }
        if (name == "--no-boundary-term") {
            options.settings.boundaryTerm = false;
            continue;
        }
        if (index + 1 >= count)
            return std::unexpected(std::format("{} needs a value", name));
        const std::string_view value = arguments[++index];
        if (name == "--algorithm") {
            options.algorithms.push_back(value);
            continue;
        }
        if (name == "--report") {
            options.report = value;
            continue;
        }
        if (name == "--dataset") {
            options.dataset = value;
            continue;
        }
        if (name == "--scenario") {
            options.scenario = value;
            continue;
        }
        if (name == "--method") {
            options.methods.push_back(value);
            continue;
        }
        if (name == "--out") {
            options.output = value;
            continue;
        }
        bool parsed = false;
        if (name == "--iterations") {
            const std::optional<int> iterations = parse<int>(value);
            parsed = iterations && *iterations > 0;
            options.iterations = iterations.value_or(0);
        } else if (name == "--seed") {
            const std::optional<std::uint64_t> seed = parse<std::uint64_t>(value);
            parsed = seed.has_value();
            options.settings.seed = seed.value_or(0);
        } else if (name == "--learning-rate-scale") {
            const std::optional<float> scale = parse<float>(value);
            parsed = scale && *scale > 0.0f;
            options.settings.learningRateScale = scale.value_or(0.0f);
        } else if (name == "--downscale") {
            const std::optional<std::uint32_t> downscale = parse<std::uint32_t>(value);
            parsed = downscale && *downscale > 0;
            options.loadOptions.downscale = downscale.value_or(0);
        } else if (name == "--seconds") {
            const std::optional<double> seconds = parse<double>(value);
            parsed = seconds && *seconds >= 0.0;
            options.budget.seconds = seconds.value_or(0.0);
        } else if (name == "--evaluate-every") {
            const std::optional<int> interval = parse<int>(value);
            parsed = interval && *interval >= 0;
            options.budget.evaluateEvery = interval.value_or(0);
        } else if (name == "--test-stride") {
            const std::optional<std::uint32_t> stride = parse<std::uint32_t>(value);
            parsed = stride && *stride > 0;
            options.budget.testStride = stride.value_or(1);
        } else {
            return std::unexpected(std::format("unknown option {}", name));
        }
        if (!parsed)
            return std::unexpected(std::format("{} has the invalid value {}", name, value));
    }
    return options;
}

std::expected<std::vector<Algorithms::Algorithm*>, std::string> select(const Algorithms::Catalog& catalog, const std::vector<std::string_view>& ids) {
    if (ids.empty() || std::ranges::find(ids, "all") != ids.end())
        return std::vector<Algorithms::Algorithm*>(catalog.getAll().begin(), catalog.getAll().end());
    std::vector<Algorithms::Algorithm*> selected;
    for (const std::string_view id : ids) {
        Algorithms::Algorithm* algorithm = catalog.find(id);
        if (!algorithm)
            return std::unexpected(std::format("unknown algorithm {}; --list shows the ids", id));
        selected.push_back(algorithm);
    }
    return selected;
}

std::expected<std::vector<Methods::Method*>, std::string> selectMethods(const Methods::Catalog& catalog, const std::vector<std::string_view>& ids,
                                                                        const Scenes::Kind kind) {
    std::vector<Methods::Method*> selected;
    if (ids.empty() || std::ranges::find(ids, "all") != ids.end()) {
        for (Methods::Method* method : catalog.getAll())
            if (method->supports(kind))
                selected.push_back(method);
    } else {
        for (const std::string_view id : ids) {
            Methods::Method* method = catalog.find(id);
            if (!method)
                return std::unexpected(std::format("unknown method {}; --list shows the ids", id));
            selected.push_back(method);
        }
    }
    if (selected.empty())
        return std::unexpected("no method of the catalog supports this scenario");
    return selected;
}

template <typename MeshMethod>
    requires std::derived_from<MeshMethod, Methods::Method>
std::expected<void, std::string> writeMesh(const MeshMethod& method, const std::filesystem::path& output, const std::string_view name) {
    const std::filesystem::path path = output / Training::Comparison::getRenderFolder(method) / name;
    if (std::expected<void, std::string> written = method.writeMesh(path); !written)
        return written;
    std::println("wrote the {} mesh to {}", method.getDescription().id, path.string());
    return {};
}

int runScenario(const Options& options) {
    const std::expected<Scenes::Dataset, std::string> dataset = Scenes::loadDataset(*options.scenario, DIFFRENDERER_DATA_ROOT, options.loadOptions);
    if (!dataset) {
        std::println(stderr, "{}", dataset.error());
        return 1;
    }
    const Methods::Catalog catalog;
    const std::expected<std::vector<Methods::Method*>, std::string> methods = selectMethods(catalog, options.methods, dataset->kind);
    if (!methods) {
        std::println(stderr, "{}", methods.error());
        return 2;
    }
    NoorRHI::Device device;
    if (std::expected<void, NoorRHI::Error> opened = device.open(); !opened) {
        std::println(stderr, "the GPU device does not open: {}", opened.error().message);
        return 1;
    }
    Training::Budget budget = options.budget;
    budget.iterations = options.iterations;
    const Methods::Settings settings{.seed = options.settings.seed,
                                     .learningRateScale = options.settings.learningRateScale,
                                     .iterations = budget.iterations,
                                     .boundaryTerm = options.settings.boundaryTerm};
    Training::Comparison comparison;
    if (std::expected<void, NoorRHI::Error> opened = comparison.open(device, *dataset, *methods, budget, settings, options.output); !opened) {
        std::println(stderr, "{}", opened.error().message);
        return 1;
    }
    std::println("{}: {} train, {} test views; {} methods", dataset->name, dataset->train.size(), dataset->test.size(), methods->size());
    while (!comparison.isFinished()) {
        const std::size_t current = comparison.getCurrentIndex();
        if (std::expected<void, NoorRHI::Error> stepped = comparison.step(); !stepped) {
            std::println(stderr, "{}", stepped.error().message);
            return 1;
        }
        if (comparison.getCurrentIndex() == current)
            continue;
        const Training::MethodResult& result = comparison.getResults()[current];
        std::println("{}: {} iterations in {:.1f} s, test PSNR {:.2f} dB", result.method->getDescription().id, result.getIterations(), result.trainSeconds,
                     result.evaluations.back().psnr);
    }
    std::print("\n{}", Training::getComparisonTable(comparison));
    if (options.output.empty())
        return 0;
    if (std::expected<void, NoorRHI::Error> written = Training::writeComparisonReport(comparison); !written) {
        std::println(stderr, "{}", written.error().message);
        return 1;
    }
    std::println("wrote the report and the renders to {}", options.output.string());
    const auto isTrained = [&](const Methods::Method& method) { return std::ranges::contains(*methods, catalog.find(method.getDescription().id)); };
    if (isTrained(catalog.getSdfGrid()))
        if (std::expected<void, std::string> written = writeMesh(catalog.getSdfGrid(), options.output, "mesh.ply"); !written) {
            std::println(stderr, "{}", written.error());
            return 1;
        }
    if (isTrained(catalog.getMeshRasterization()))
        if (std::expected<void, std::string> written = writeMesh(catalog.getMeshRasterization(), options.output, "mesh.obj"); !written) {
            std::println(stderr, "{}", written.error());
            return 1;
        }
    return 0;
}

void drawPoints(const Scenes::PointCloud& points, const Scenes::Camera& camera, Core::Image& image) {
    const std::span<float> pixels = image.getPixels();
    const auto& [first, second, third] = camera.worldToCamera;
    for (std::size_t index = 0; index < points.positions.size(); ++index) {
        const NoorRHI::float3 p = points.positions[index];
        const float depth = third.x * p.x + third.y * p.y + third.z * p.z + third.w;
        if (depth <= 0.0f)
            continue;
        const float u = camera.intrinsics.x * (first.x * p.x + first.y * p.y + first.z * p.z + first.w) / depth + camera.intrinsics.z;
        const float v = camera.intrinsics.y * (second.x * p.x + second.y * p.y + second.z * p.z + second.w) / depth + camera.intrinsics.w;
        if (u < 0.0f || v < 0.0f || u >= static_cast<float>(camera.width) || v >= static_cast<float>(camera.height))
            continue;
        const std::size_t pixel = static_cast<std::size_t>(v) * camera.width + static_cast<std::size_t>(u);
        const NoorRHI::float3 color = points.colors[index];
        pixels[pixel * 3] = color.x;
        pixels[pixel * 3 + 1] = color.y;
        pixels[pixel * 3 + 2] = color.z;
    }
}

int checkDataset(const std::string_view id, const Scenes::LoadOptions& loadOptions) {
    const std::filesystem::path dataRoot = DIFFRENDERER_DATA_ROOT;
    const std::expected<Scenes::Dataset, std::string> dataset = Scenes::loadDataset(id, dataRoot, loadOptions);
    if (!dataset) {
        std::println(stderr, "{}", dataset.error());
        return 1;
    }
    const Scenes::View& view = dataset->train.front();
    const NoorRHI::float3 center = dataset->center;
    std::println("{}: {} train, {} test views of {}x{}, {} points, centre ({:.3f}, {:.3f}, {:.3f}), extent {:.3f}", dataset->name, dataset->train.size(),
                 dataset->test.size(), view.image.getWidth(), view.image.getHeight(), dataset->points.positions.size(), center.x, center.y, center.z,
                 dataset->extent);
    const std::filesystem::path imagePath = std::filesystem::path("/var/tmp") / std::format("{}-train.png", id);
    Core::Image overlay = view.image;
    drawPoints(dataset->points, view.camera, overlay);
    const std::filesystem::path overlayPath = std::filesystem::path("/var/tmp") / std::format("{}-points.png", id);
    const std::array<std::pair<const Core::Image*, std::filesystem::path>, 2> outputs{{{&view.image, imagePath}, {&overlay, overlayPath}}};
    for (const auto& [image, path] : outputs)
        if (std::expected<void, std::string> saved = Core::savePng(*image, path); !saved) {
            std::println(stderr, "{}", saved.error());
            return 1;
        }
    std::println("wrote {} and {}", imagePath.string(), overlayPath.string());
    return 0;
}

}

int main(const int count, char** arguments) {
    const std::expected<Options, std::string> options = parseOptions(count, arguments);
    if (!options) {
        std::println(stderr, "{}\n{}", options.error(), usage);
        return 2;
    }
    if (options->dataset)
        return checkDataset(*options->dataset, options->loadOptions);
    if (options->scenario)
        return runScenario(*options);
    Algorithms::Catalog catalog;
    if (options->list) {
        for (const Algorithms::Algorithm* algorithm : catalog.getAll()) {
            const Algorithms::Description& description = algorithm->getDescription();
            std::println("{:<22} {} ({}), reference: {}", description.id, description.name, description.representation, description.reference);
        }
        for (const Methods::Method* method : Methods::Catalog().getAll()) {
            const Methods::Description& description = method->getDescription();
            std::println("method {:<15} {} ({} + {}), reference: {}", description.id, description.name, description.representation, description.renderer,
                         description.reference);
        }
        return 0;
    }
    const std::expected<std::vector<Algorithms::Algorithm*>, std::string> selected = select(catalog, options->algorithms);
    if (!selected) {
        std::println(stderr, "{}", selected.error());
        return 2;
    }
    NoorRHI::Device device;
    if (std::expected<void, NoorRHI::Error> opened = device.open(); !opened) {
        std::println(stderr, "the GPU device does not open: {}", opened.error().message);
        return 1;
    }
    std::vector<Training::Run> runs(selected->size());
    std::vector<const Training::Run*> finished;
    for (std::size_t index = 0; index < runs.size(); ++index) {
        Algorithms::Algorithm& algorithm = *(*selected)[index];
        Training::Run& run = runs[index];
        std::println("{}: {} iterations", algorithm.getDescription().name, options->iterations);
        if (std::expected<void, NoorRHI::Error> opened = run.open(algorithm, device, options->settings); !opened) {
            std::println(stderr, "{} does not open: {}", algorithm.getDescription().id, opened.error().message);
            return 1;
        }
        for (int iteration = 0; iteration < options->iterations; ++iteration)
            if (std::expected<void, NoorRHI::Error> stepped = run.step(); !stepped) {
                std::println(stderr, "{} failed at iteration {}: {}", algorithm.getDescription().id, iteration, stepped.error().message);
                return 1;
            }
        finished.push_back(&run);
    }
    std::print("\n{}", Training::getReportTable(finished));
    if (!options->report)
        return 0;
    if (std::expected<void, NoorRHI::Error> written = Training::writeReport(*options->report, finished); !written) {
        std::println(stderr, "{}", written.error().message);
        return 1;
    }
    std::println("wrote {}", options->report->string());
    return 0;
}
