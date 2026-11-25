#include "NoorRay.Cli/DlssRrFixture.h"

#include <format>
#include <fstream>
#include <span>
#include <string_view>
#include <system_error>

namespace NoorRay::Cli {

namespace {

std::string_view getFormatName(const NoorRHI::ImageFormat format) {
    switch (format) {
    case NoorRHI::ImageFormat::Rgba16Float:
        return "R16G16B16A16_SFLOAT";
    case NoorRHI::ImageFormat::Rg16Float:
        return "R16G16_SFLOAT";
    case NoorRHI::ImageFormat::R16Float:
        return "R16_SFLOAT";
    case NoorRHI::ImageFormat::R32Float:
        return "R32_SFLOAT";
    default:
        return "UNDEFINED";
    }
}

std::string getMatrix(const std::array<float, 16>& matrix) {
    std::string text = "[";
    for (std::size_t index = 0; index < matrix.size(); ++index)
        text += std::format("{}{}", index == 0 ? "" : ", ", matrix[index]);
    return text + "]";
}

std::expected<void, Core::Error> writeFile(const std::filesystem::path& path, const std::span<const std::byte> bytes) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!file)
        return Core::fail(Core::ErrorCode::Io, "cannot write " + path.string());
    return {};
}

std::expected<void, Core::Error> writeImage(const std::filesystem::path& path, const NoorRHI::Image& image) {
    std::vector<std::byte> bytes(image.getByteSize());
    if (auto downloaded = Core::check(image.download(bytes)); !downloaded)
        return downloaded;
    return writeFile(path, bytes);
}

}

std::expected<void, Core::Error> DlssRrFixture::open(const std::filesystem::path& opened) {
    std::error_code error;
    std::filesystem::create_directories(opened, error);
    if (error)
        return Core::fail(Core::ErrorCode::Io, std::format("cannot create {}: {}", opened.string(), error.message()));
    directory = opened;
    frames.clear();
    return {};
}

std::expected<void, Core::Error> DlssRrFixture::addFrame(const Render::DlssRr& dlssRr) {
    const std::filesystem::path frameDirectory = directory / std::format("frame_{:04}", frames.size());
    std::error_code error;
    std::filesystem::create_directories(frameDirectory, error);
    if (error)
        return Core::fail(Core::ErrorCode::Io, std::format("cannot create {}: {}", frameDirectory.string(), error.message()));
    for (const Render::DlssRrInput& input : dlssRr.getInputs())
        if (auto written = writeImage(frameDirectory / (std::string(input.name) + ".bin"), *input.image); !written)
            return written;
    if (auto written = writeImage(frameDirectory / "output.bin", dlssRr.getOutput()); !written)
        return written;
    const Render::DlssRrEvaluation& evaluation = dlssRr.getEvaluation();
    frames.push_back(std::format("    {{\n"
                                 "      \"jitter\": [{}, {}],\n"
                                 "      \"renderSubrectDimensions\": [{}, {}],\n"
                                 "      \"mvScale\": [{}, {}],\n"
                                 "      \"reset\": {},\n"
                                 "      \"preExposure\": {},\n"
                                 "      \"exposureScale\": {},\n"
                                 "      \"frameTimeDeltaMs\": {},\n"
                                 "      \"worldToView\": {},\n"
                                 "      \"viewToClip\": {}\n"
                                 "    }}",
                                 evaluation.jitter.x, evaluation.jitter.y, evaluation.renderSubrect.x, evaluation.renderSubrect.y, evaluation.mvScale.x,
                                 evaluation.mvScale.y, evaluation.reset, evaluation.preExposure, evaluation.exposureScale, evaluation.frameTimeDeltaMs,
                                 getMatrix(evaluation.worldToView), getMatrix(evaluation.viewToClip)));
    return {};
}

std::expected<void, Core::Error> DlssRrFixture::close(const Render::DlssRr& dlssRr) {
    const Render::DlssRrCreation& creation = dlssRr.getCreation();
    std::string inputs;
    for (const Render::DlssRrInput& input : dlssRr.getInputs())
        inputs += std::format("{}    \"{}\": {{ \"format\": \"{}\" }}", inputs.empty() ? "" : ",\n", input.name, getFormatName(input.image->getFormat()));
    std::string frameList;
    for (const std::string& frame : frames)
        frameList += (frameList.empty() ? "" : ",\n") + frame;
    const std::string text = std::format("{{\n"
                                         "  \"library\": \"{}\",\n"
                                         "  \"renderWidth\": {}, \"renderHeight\": {},\n"
                                         "  \"outputWidth\": {}, \"outputHeight\": {},\n"
                                         "  \"inputs\": {{\n{}\n  }},\n"
                                         "  \"output\": {{ \"format\": \"{}\" }},\n"
                                         "  \"frames\": [\n{}\n  ]\n"
                                         "}}\n",
                                         dlssRr.getLibrary(), creation.renderWidth, creation.renderHeight, creation.outputWidth, creation.outputHeight, inputs,
                                         getFormatName(dlssRr.getOutput().getFormat()), frameList);
    return writeFile(directory / "sequence.json", std::as_bytes(std::span(text)));
}

}
