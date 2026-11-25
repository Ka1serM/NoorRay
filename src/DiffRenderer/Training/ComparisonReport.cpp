#include "DiffRenderer/Training/ComparisonReport.h"

#include <format>
#include <fstream>
#include <limits>

#include "DiffRenderer/Training/Json.h"

namespace DiffRenderer::Training {

namespace {

constexpr double bytesPerMegabyte = 1024.0 * 1024.0;

std::string_view getKindName(const Scenes::Kind kind) {
    switch (kind) {
    case Scenes::Kind::Scene:
        return "scene";
    case Scenes::Kind::Object:
        return "object";
    case Scenes::Kind::Volume:
        return "volume";
    }
    return "";
}

double getFinal(const MethodResult& result, double Evaluation::* metric) {
    if (!result.finished || result.evaluations.empty())
        return std::numeric_limits<double>::quiet_NaN();
    return result.evaluations.back().*metric;
}

template <class T>
std::string toJsonArray(const std::vector<T>& values) {
    std::string text;
    for (const T value : values)
        text += std::format("{}{}", text.empty() ? "" : ", ", toJson(value));
    return text;
}

std::string describe(const MethodResult& result) {
    const Methods::Description& description = result.method->getDescription();
    std::string evaluations;
    for (const Evaluation& evaluation : result.evaluations)
        evaluations += std::format("{}[{}, {}, {}, {}]", evaluations.empty() ? "" : ", ", evaluation.iteration, toJson(evaluation.trainSeconds),
                                   toJson(evaluation.psnr), toJson(evaluation.ssim));
    std::string views;
    for (std::size_t index = 0; index < result.views.size(); ++index) {
        const ViewResult& view = result.views[index];
        views +=
            std::format("{}\n        {{\"name\": {}, \"file\": {}, \"psnr\": {}, \"ssim\": {}, \"render_ms\": {}}}", views.empty() ? "" : ",", quote(view.name),
                        quote(Comparison::getImageName(index).string()), toJson(view.psnr), toJson(view.ssim), toJson(view.renderMilliseconds));
    }
    return std::format("    {{\n      \"id\": {},\n      \"name\": {},\n      \"reference\": {},\n      \"finished\": {},\n      \"iterations\": {},\n"
                       "      \"train_seconds\": {},\n      \"open_ms\": {},\n      \"median_step_ms\": {},\n      \"psnr\": {},\n      \"ssim\": {},\n"
                       "      \"lpips\": null,\n      \"render_ms\": {},\n      \"primitives\": {},\n      \"bytes\": {},\n      \"renders\": {},\n"
                       "      \"evaluation_columns\": [\"iteration\", \"train_seconds\", \"psnr\", \"ssim\"],\n      \"evaluations\": [{}],\n"
                       "      \"views\": [{}\n      ],\n      \"losses\": [{}],\n      \"step_ms\": [{}]\n    }}",
                       quote(description.id), quote(description.name), quote(description.reference), result.finished ? "true" : "false", result.getIterations(),
                       toJson(result.trainSeconds), toJson(result.openMilliseconds), toJson(result.getMedianStepMilliseconds()),
                       toJson(getFinal(result, &Evaluation::psnr)), toJson(getFinal(result, &Evaluation::ssim)), toJson(result.getMeanRenderMilliseconds()),
                       result.statistics.primitives, result.statistics.bytes, quote(Comparison::getRenderFolder(*result.method).string()), evaluations, views,
                       toJsonArray(result.losses), toJsonArray(result.stepMilliseconds));
}

}

std::string getComparisonJson(const Comparison& comparison) {
    const Scenes::Dataset& dataset = comparison.getDataset();
    const Budget& budget = comparison.getBudget();
    std::string methods;
    for (const MethodResult& result : comparison.getResults())
        methods += (methods.empty() ? "" : ",\n") + describe(result);
    return std::format("{{\n  \"dataset\": {{\"id\": {}, \"name\": {}, \"kind\": {}, \"train_views\": {}, \"test_views\": {}}},\n"
                       "  \"budget\": {{\"iterations\": {}, \"seconds\": {}, \"evaluate_every\": {}, \"test_stride\": {}}},\n"
                       "  \"seed\": {},\n  \"loss\": {},\n  \"references\": {},\n  \"methods\": [\n{}\n  ]\n}}\n",
                       quote(dataset.id), quote(dataset.name), quote(getKindName(dataset.kind)), dataset.train.size(), dataset.test.size(), budget.iterations,
                       toJson(budget.seconds), budget.evaluateEvery, budget.testStride, comparison.getSettings().seed,
                       quote(comparison.getLossKind() == LossKind::L2 ? "l2" : "l1_ssim"), quote(Comparison::getReferenceFolder().string()), methods);
}

std::string getComparisonTable(const Comparison& comparison) {
    std::string table = "| Method | Iterations | Train s | Test PSNR (dB) | Test SSIM | ms / step | Render ms | Primitives | Memory MB |\n"
                        "|---|---|---|---|---|---|---|---|---|\n";
    for (const MethodResult& result : comparison.getResults())
        table += std::format("| {} | {} | {:.1f} | {:.2f} | {:.4f} | {:.2f} | {:.2f} | {} | {:.1f} |\n", result.method->getDescription().name,
                             result.getIterations(), result.trainSeconds, getFinal(result, &Evaluation::psnr), getFinal(result, &Evaluation::ssim),
                             result.getMedianStepMilliseconds(), result.getMeanRenderMilliseconds(), result.statistics.primitives,
                             static_cast<double>(result.statistics.bytes) / bytesPerMegabyte);
    return table;
}

std::expected<void, NoorRHI::Error> writeComparisonReport(const Comparison& comparison) {
    for (const auto& [name, text] : {std::pair{"report.json", getComparisonJson(comparison)}, std::pair{"report.md", getComparisonTable(comparison)}}) {
        const std::filesystem::path path = comparison.getOutput() / name;
        std::ofstream file(path);
        if (!file)
            return NoorRHI::fail(NoorRHI::ErrorCode::InvalidArgument, std::format("cannot write {}", path.string()));
        file << text;
    }
    return {};
}

}
