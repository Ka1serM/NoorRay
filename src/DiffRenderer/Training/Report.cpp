#include "DiffRenderer/Training/Report.h"

#include <format>
#include <fstream>

#include "DiffRenderer/Training/Json.h"

namespace DiffRenderer::Training {

namespace {

std::string describe(const Run& run) {
    const Algorithms::Description& description = run.getAlgorithm().getDescription();
    const Algorithms::Settings& settings = run.getSettings();
    std::string history;
    for (const Sample& sample : run.getSamples())
        history += std::format("{}[{}, {}, {}, {}, {}]", history.empty() ? "" : ", ", sample.iteration, toJson(sample.loss), toJson(sample.psnr),
                               toJson(sample.parameterError), toJson(sample.milliseconds));
    return std::format("    {{\n      \"id\": {},\n      \"name\": {},\n      \"reference\": {},\n      \"seed\": {},\n      \"learning_rate_scale\": {},\n"
                       "      \"boundary_term\": {},\n      \"open_ms\": {},\n      \"median_step_ms\": {},\n"
                       "      \"history_columns\": [\"iteration\", \"loss\", \"psnr\", \"parameter_error\", \"ms\"],\n      \"history\": [{}]\n    }}",
                       quote(description.id), quote(description.name), quote(description.reference), settings.seed, toJson(settings.learningRateScale),
                       description.hasBoundaryTerm && settings.boundaryTerm ? "true" : "false", toJson(run.getOpenMilliseconds()),
                       toJson(run.getMedianStepMilliseconds()), history);
}

}

std::string getReportJson(const std::span<const Run* const> runs) {
    std::string entries;
    for (const Run* run : runs)
        entries += (entries.empty() ? "" : ",\n") + describe(*run);
    return std::format("{{\n  \"runs\": [\n{}\n  ]\n}}\n", entries);
}

std::string getReportTable(const std::span<const Run* const> runs) {
    std::string table = "| Algorithm | Iterations | Loss start | Loss end | PSNR end (dB) | Parameter error | ms / step |\n|---|---|---|---|---|---|---|\n";
    for (const Run* run : runs) {
        const std::span<const Sample> samples = run->getSamples();
        if (samples.empty())
            continue;
        table += std::format("| {} | {} | {:.3e} | {:.3e} | {:.2f} | {:.4g} | {:.2f} |\n", run->getAlgorithm().getDescription().name, samples.size(),
                             samples.front().loss, samples.back().loss, samples.back().psnr, samples.back().parameterError, run->getMedianStepMilliseconds());
    }
    return table;
}

std::expected<void, NoorRHI::Error> writeReport(const std::filesystem::path& path, const std::span<const Run* const> runs) {
    std::ofstream file(path);
    if (!file)
        return NoorRHI::fail(NoorRHI::ErrorCode::InvalidArgument, std::format("cannot write {}", path.string()));
    file << getReportJson(runs);
    return {};
}

}
