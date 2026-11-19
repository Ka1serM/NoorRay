#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <expected>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <tinyexr.h>

#include "NoorRHI/Device/Device.h"
#include "NoorRHI/Resources/Readback.h"
#include "NoorRay/Session/MaterialPreviews.h"
#include "NoorRay/Session/Session.h"

namespace {

using NoorRay::Core::Error;
using NoorRay::Core::ErrorCode;

struct Options {
    std::filesystem::path scene;
    //! With --previews, the output is a folder that gets one preview per material of the scene.
    bool previews = false;
    std::filesystem::path output;
    std::uint32_t samples = 64;
    std::uint32_t width = 1280;
    std::uint32_t height = 720;
    bool restir = false;
};

std::expected<std::uint32_t, Error> parseNumber(const std::string_view text) {
    std::uint32_t value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || value == 0)
        return NoorRay::Core::fail(ErrorCode::InvalidArgument, "Not a positive number: " + std::string(text));
    return value;
}

std::expected<Options, Error> parseOptions(std::span<char*> arguments) {
    if (arguments.size() == 4 && std::string_view(arguments[1]) == "--previews")
        return Options{.scene = arguments[2], .previews = true, .output = arguments[3]};
    const bool restir = arguments.size() > 1 && std::string_view(arguments.back()) == "--restir";
    if (restir)
        arguments = arguments.first(arguments.size() - 1);
    if (arguments.size() != 3 && arguments.size() != 4 && arguments.size() != 6)
        return NoorRay::Core::fail(ErrorCode::InvalidArgument,
                                   "Usage: noorray <scene> <output.exr> [samples [width height]] [--restir]\n       noorray --previews <scene> <folder>");
    Options options{.scene = arguments[1], .output = arguments[2], .restir = restir};
    if (arguments.size() == 3)
        return options;
    auto samples = parseNumber(arguments[3]);
    if (!samples)
        return std::unexpected(samples.error());
    options.samples = *samples;
    if (arguments.size() == 4)
        return options;
    auto width = parseNumber(arguments[4]);
    if (!width)
        return std::unexpected(width.error());
    auto height = parseNumber(arguments[5]);
    if (!height)
        return std::unexpected(height.error());
    options.width = *width;
    options.height = *height;
    return options;
}

std::expected<void, Error> submitFrame(NoorRHI::Device& device, NoorRHI::CommandList&& list) {
    auto token = NoorRay::Core::check(device.submit(std::move(list)));
    if (!token)
        return std::unexpected(token.error());
    return NoorRay::Core::check(device.wait(*token));
}

std::expected<void, Error> writeExr(const std::filesystem::path& path, const std::span<const NoorRHI::float4> pixels, const std::uint32_t width,
                                    const std::uint32_t height) {
    std::vector<float> rgba;
    rgba.reserve(pixels.size() * 4);
    for (const NoorRHI::float4& pixel : pixels)
        rgba.insert(rgba.end(), {pixel.x, pixel.y, pixel.z, pixel.w});
    const char* message = nullptr;
    if (SaveEXR(rgba.data(), static_cast<int>(width), static_cast<int>(height), 4, 0, path.string().c_str(), &message) == TINYEXR_SUCCESS)
        return {};
    std::string text = message ? message : "SaveEXR failed";
    FreeEXRErrorMessage(message);
    return NoorRay::Core::fail(ErrorCode::Io, text);
}

std::expected<void, Error> openDevice(NoorRHI::Device& device) {
    return NoorRay::Core::check(device.open({.enableValidation = std::getenv("NOORRAY_VALIDATION") != nullptr, .applicationName = "NoorRay"}));
}

//! Renders the material previews of a scene the way the app does and writes each film as <index>-<material name>.exr.
std::expected<void, Error> renderPreviews(const Options& options) {
    NoorRHI::Device device;
    if (auto opened = openDevice(device); !opened)
        return opened;
    NoorRay::Session::Session session;
    if (auto opened = session.open(device, 1, 1); !opened)
        return opened;
    if (auto imported = session.importFile(options.scene); !imported)
        return imported;
    NoorRay::Session::MaterialPreviews previews;
    if (auto opened = previews.open(device, session.getRenderer().getPipelines()); !opened)
        return opened;
    const NoorRay::Scene::Scene& scene = session.getScene();
    std::vector<NoorRHI::Readback<NoorRHI::float4>> films(scene.getMaterials().size());
    std::vector<bool> written(films.size(), false);
    std::expected<void, Error> failure;
    previews.setFinished([&](const NoorRay::Scene::MaterialHandle material, NoorRHI::CommandList& list, const NoorRay::Render::Renderer& renderer) {
        const NoorRHI::Buffer<NoorRHI::float4>& output = renderer.getTargets().getOutput();
        for (std::uint32_t i = 0; i < films.size(); ++i)
            if (scene.getMaterials().keyAt(i) == material) {
                if (auto opened = NoorRay::Core::check(films[i].open(device, output.getSize())); !opened)
                    failure = opened;
                else
                    list.download(output, films[i]);
            }
    });
    for (std::uint32_t i = 0; i < films.size(); ++i)
        static_cast<void>(previews.request(scene, scene.getMaterials().keyAt(i)));
    std::filesystem::create_directories(options.output);
    while (previews.isBusy()) {
        NoorRHI::CommandList list(device);
        if (auto recorded = previews.record(list, scene); !recorded)
            return std::unexpected(recorded.error());
        if (auto submitted = submitFrame(device, std::move(list)); !submitted)
            return submitted;
        if (!failure)
            return failure;
    }
    const std::uint32_t size = NoorRay::Session::MaterialPreviews::defaultSize;
    for (std::uint32_t i = 0; i < films.size(); ++i) {
        if (!films[i].getBackend())
            continue;
        auto pixels = NoorRay::Core::check(films[i].read());
        if (!pixels)
            return std::unexpected(pixels.error());
        std::string name = scene.getMaterials().getValues()[i].name;
        std::ranges::replace_if(name, [](const char character) { return character == '/' || character == '\\' || character == ':'; }, '_');
        const std::filesystem::path path = options.output / (std::to_string(i) + "-" + name + ".exr");
        if (auto written = writeExr(path, *pixels, size, size); !written)
            return written;
        std::printf("%s\n", path.string().c_str());
    }
    return {};
}

std::expected<void, Error> render(const Options& options) {
    NoorRHI::Device device;
    if (auto opened = openDevice(device); !opened)
        return opened;
    NoorRay::Session::Session session;
    if (auto opened = session.open(device, options.width, options.height); !opened)
        return opened;
    if (auto imported = session.importFile(options.scene); !imported)
        return imported;
    NoorRay::Render::RendererSettings settings = NoorRay::Render::RendererSettings::pathTraced();
    settings.maxSamples = options.samples;
    settings.wavefront.compilesInBackground = false;
    settings.reuse = options.restir ? NoorRay::Render::Reuse::DiPt : NoorRay::Render::Reuse::Off;
    session.getRenderer().setSettings(settings);
    while (!session.getRenderer().isComplete()) {
        NoorRHI::CommandList list(device);
        if (auto recorded = session.record(list); !recorded)
            return recorded;
        if (auto submitted = submitFrame(device, std::move(list)); !submitted)
            return submitted;
    }
    const NoorRHI::Buffer<NoorRHI::float4>& output = session.getRenderer().getTargets().getOutput();
    NoorRHI::Readback<NoorRHI::float4> readback;
    if (auto opened = NoorRay::Core::check(readback.open(device, output.getSize())); !opened)
        return opened;
    NoorRHI::CommandList list(device);
    list.download(output, readback);
    if (auto submitted = submitFrame(device, std::move(list)); !submitted)
        return submitted;
    auto pixels = NoorRay::Core::check(readback.read());
    if (!pixels)
        return std::unexpected(pixels.error());
    return writeExr(options.output, *pixels, options.width, options.height);
}

}

int main(int argc, char** argv) {
    auto options = parseOptions(std::span(argv, static_cast<std::size_t>(argc)));
    std::expected<void, Error> result =
        options ? (options->previews ? renderPreviews(*options) : render(*options)) : std::expected<void, Error>(std::unexpected(options.error()));
    if (result)
        return 0;
    std::fprintf(stderr, "%s\n", result.error().message.c_str());
    return 1;
}
