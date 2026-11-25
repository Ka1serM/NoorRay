#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <expected>
#include <filesystem>
#include <format>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <tinyexr.h>

#include "NoorRHI/Commands/TimestampQuery.h"
#include "NoorRHI/Device/Device.h"
#include "NoorRHI/Resources/Readback.h"
#include "NoorRay.Cli/DlssRrFixture.h"
#include "NoorRay/Core/CacheDirectory.h"
#include "NoorRay/Render/Denoise/DenoiserPlugin.h"
#include "NoorRay/Render/GpuProfiler.h"
#include "NoorRay/Session/MaterialPreviews.h"
#include "NoorRay/Session/Session.h"

namespace {

using NoorRay::Core::Error;
using NoorRay::Core::ErrorCode;

struct FrameRange {
    int first = 0;
    int last = 0;
};

struct Options {
    std::filesystem::path scene;
    //! With --previews, the output is a folder that gets one preview per material of the scene.
    bool previews = false;
    std::filesystem::path output;
    std::uint32_t samples = 64;
    std::uint32_t width = 1280;
    std::uint32_t height = 720;
    //! --reuse off|di|pt picks no reuse, ReSTIR DI or ReSTIR PT.
    NoorRay::Render::Reuse reuse = NoorRay::Render::Reuse::Off;
    //! --gaussians off|emissive|scattering leaves the Gaussians out of the rays, shows their colour, or lights them.
    NoorRay::Render::GaussianShading gaussianShading = NoorRay::Render::GaussianShading::Emissive;
    bool denoise = true;
    //! --upscale renders at half size for FSR.
    bool upscale = false;
    //! --no-temporal keeps each frame apart, so it excludes --upscale.
    bool temporal = true;
    //! --no-cache turns the SHaRC radiance cache off.
    bool cache = true;
    //! --timings prints the GPU time of each frame and a summary; --profile adds the time of each kernel, which waits between kernels.
    bool timings = false;
    bool profile = false;
    std::optional<std::uint32_t> depth;
    std::optional<std::uint32_t> pool;
    //! --scale S renders at S times the output size and implies --upscale, or scales --open-dlss-rr.
    std::optional<float> scale;
    //! --open-dlss-rr renders at half size for OpenDLSS-RR, which replaces NRD and FSR.
    bool dlssRr = false;
    //! --open-dlss-rr-dump D implies --open-dlss-rr and writes the inputs and output of each of the samples frames as a fixture in D.
    std::optional<std::filesystem::path> dlssRrDump;
    //! --exact-volumes runs the volume density graphs at each point instead of their sparse grids.
    bool exactVolumes = false;
    //! --clamp C scales each indirect sample down to a component sum of C, as Cycles' indirect clamp does; 0 turns it off.
    float clamp = 0.0f;
    //! With --frames, each # run in the output name becomes the zero-padded frame number.
    std::optional<FrameRange> frames;
};

constexpr std::string_view usage =
    "Usage: noorray <scene> <output.exr> [samples [width height]] [--reuse off|di|pt] [--gaussians off|emissive|scattering] [--no-denoise] [--upscale] "
    "[--no-temporal] [--no-cache] [--timings | --profile] [--depth N] [--pool N] [--scale S] [--clamp C] [--exact-volumes]\n"
    "       [--frames N | --frames A-B] [--open-dlss-rr] [--open-dlss-rr-dump folder]\n"
    "       noorray --previews <scene> <folder>";

std::expected<std::uint32_t, Error> parseNumber(const std::string_view text) {
    std::uint32_t value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || value == 0)
        return NoorRay::Core::fail(ErrorCode::InvalidArgument, "Not a positive number: " + std::string(text));
    return value;
}

std::expected<int, Error> parseFrame(const std::string_view text) {
    int value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size())
        return NoorRay::Core::fail(ErrorCode::InvalidArgument, "Not a frame number: " + std::string(text));
    return value;
}

std::expected<FrameRange, Error> parseFrames(const std::string_view text) {
    const std::size_t dash = text.find('-', 1);
    auto first = parseFrame(text.substr(0, dash));
    if (!first)
        return std::unexpected(first.error());
    if (dash == std::string_view::npos)
        return FrameRange{*first, *first};
    auto last = parseFrame(text.substr(dash + 1));
    if (!last)
        return std::unexpected(last.error());
    if (*last < *first)
        return NoorRay::Core::fail(ErrorCode::InvalidArgument, "The frame range ends before it starts: " + std::string(text));
    return FrameRange{*first, *last};
}

std::filesystem::path getFramePath(const std::filesystem::path& pattern, const int frame) {
    std::string name = pattern.filename().string();
    const std::size_t first = name.find('#');
    if (first == std::string::npos)
        return pattern;
    const std::size_t end = name.find_first_not_of('#', first);
    const std::size_t count = (end == std::string::npos ? name.size() : end) - first;
    name.replace(first, count, std::format("{:0{}}", frame, count));
    return pattern.parent_path() / name;
}

std::expected<Options, Error> parseOptions(const std::span<char*> arguments) {
    if (arguments.size() == 4 && std::string_view(arguments[1]) == "--previews")
        return Options{.scene = arguments[2], .previews = true, .output = arguments[3]};
    Options options;
    std::vector<std::string_view> positional;
    for (std::size_t index = 1; index < arguments.size(); ++index) {
        const std::string_view argument = arguments[index];
        if (argument == "--reuse" && index + 1 < arguments.size()) {
            const std::string_view text = arguments[++index];
            if (text == "off")
                options.reuse = NoorRay::Render::Reuse::Off;
            else if (text == "di")
                options.reuse = NoorRay::Render::Reuse::Di;
            else if (text == "pt")
                options.reuse = NoorRay::Render::Reuse::Pt;
            else
                return NoorRay::Core::fail(ErrorCode::InvalidArgument, "--reuse needs off, di or pt: " + std::string(text));
            continue;
        }
        if (argument == "--gaussians" && index + 1 < arguments.size()) {
            const std::string_view text = arguments[++index];
            if (text == "off")
                options.gaussianShading = NoorRay::Render::GaussianShading::Off;
            else if (text == "emissive")
                options.gaussianShading = NoorRay::Render::GaussianShading::Emissive;
            else if (text == "scattering")
                options.gaussianShading = NoorRay::Render::GaussianShading::Scattering;
            else
                return NoorRay::Core::fail(ErrorCode::InvalidArgument, "--gaussians needs off, emissive or scattering: " + std::string(text));
            continue;
        }
        if (argument == "--no-denoise") {
            options.denoise = false;
            continue;
        }
        if (argument == "--no-temporal") {
            options.temporal = false;
            continue;
        }
        if (argument == "--no-cache") {
            options.cache = false;
            continue;
        }
        if (argument == "--exact-volumes") {
            options.exactVolumes = true;
            continue;
        }
        if (argument == "--timings" || argument == "--profile") {
            options.timings = true;
            options.profile = options.profile || argument == "--profile";
            continue;
        }
        if ((argument == "--depth" || argument == "--pool") && index + 1 < arguments.size()) {
            auto value = parseNumber(arguments[++index]);
            if (!value)
                return std::unexpected(value.error());
            (argument == "--depth" ? options.depth : options.pool) = *value;
            continue;
        }
        if (argument == "--scale" && index + 1 < arguments.size()) {
            const std::string_view text = arguments[++index];
            float value = 0.0f;
            const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
            if (error != std::errc{} || end != text.data() + text.size() || value < 0.25f || value > 1.0f)
                return NoorRay::Core::fail(ErrorCode::InvalidArgument, "--scale needs a number from 0.25 to 1: " + std::string(text));
            options.scale = value;
            continue;
        }
        if (argument == "--open-dlss-rr" || (argument == "--open-dlss-rr-dump" && index + 1 < arguments.size())) {
            options.dlssRr = true;
            if (argument == "--open-dlss-rr-dump")
                options.dlssRrDump = arguments[++index];
            continue;
        }
        if (argument == "--clamp" && index + 1 < arguments.size()) {
            const std::string_view text = arguments[++index];
            float value = 0.0f;
            const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
            if (error != std::errc{} || end != text.data() + text.size() || value < 0.0f)
                return NoorRay::Core::fail(ErrorCode::InvalidArgument, "--clamp needs a number of at least 0: " + std::string(text));
            options.clamp = value;
            continue;
        }
        if (argument == "--upscale") {
            options.upscale = true;
            continue;
        }
        if (argument != "--frames") {
            positional.push_back(argument);
            continue;
        }
        if (++index == arguments.size())
            return NoorRay::Core::fail(ErrorCode::InvalidArgument, std::string(usage));
        auto frames = parseFrames(arguments[index]);
        if (!frames)
            return std::unexpected(frames.error());
        options.frames = *frames;
    }
    options.upscale = options.upscale || (options.scale && !options.dlssRr);
    if (options.upscale && !options.temporal)
        return NoorRay::Core::fail(ErrorCode::InvalidArgument, "--upscale needs temporal history, so it cannot go with --no-temporal");
    if (options.dlssRr && (options.upscale || !options.temporal || !options.denoise))
        return NoorRay::Core::fail(ErrorCode::InvalidArgument, "--open-dlss-rr denoises and upscales with temporal history, so it cannot go with --upscale, "
                                                               "--no-temporal or --no-denoise");
    if (positional.size() != 2 && positional.size() != 3 && positional.size() != 5)
        return NoorRay::Core::fail(ErrorCode::InvalidArgument, std::string(usage));
    options.scene = positional[0];
    options.output = positional[1];
    if (options.frames && options.frames->first != options.frames->last && options.output.filename().string().find('#') == std::string::npos)
        return NoorRay::Core::fail(ErrorCode::InvalidArgument, "A frame range needs # in the output name, for example out_####.exr");
    if (positional.size() == 2)
        return options;
    auto samples = parseNumber(positional[2]);
    if (!samples)
        return std::unexpected(samples.error());
    options.samples = *samples;
    if (positional.size() == 3)
        return options;
    auto width = parseNumber(positional[3]);
    if (!width)
        return std::unexpected(width.error());
    auto height = parseNumber(positional[4]);
    if (!height)
        return std::unexpected(height.error());
    options.width = *width;
    options.height = *height;
    return options;
}

//! A realtime frame samples each pixel once, so only FSR or DLSS-RR anti-aliases it; without them the CLI accumulates, as a final render does.
bool rendersRealtime(const Options& options) {
    return options.upscale || options.dlssRr || std::getenv("NR_RT") != nullptr;
}

bool resolvesTemporally(const Options& options) {
    return (options.upscale || options.dlssRr) && options.temporal;
}

std::expected<void, Error> submitFrame(NoorRHI::Device& device, NoorRHI::CommandList&& list) {
    auto token = NoorRay::Core::check(device.submit(std::move(list)));
    if (!token)
        return std::unexpected(token.error());
    return NoorRay::Core::check(device.wait(*token));
}

//! The first frames compile and fill the history, so the summary leaves out the first quarter.
std::size_t steadyFrame(const std::uint32_t frames) {
    return frames / 4;
}

void printTimings(const std::vector<double>& times, const NoorRay::Render::GpuProfiler& profiler) {
    for (std::size_t index = 0; index < times.size(); ++index)
        std::printf("frame %zu: %.2f ms\n", index, times[index]);
    const double frames = std::max(profiler.getFrameCount(), 1u);
    for (const NoorRay::Render::GpuProfiler::Total& total : profiler.getTotals())
        std::printf("  %-28s %8.3f ms %6.1f calls per frame\n", total.name.c_str(), total.milliseconds / frames, total.calls / frames);
    std::vector<double> steady(times.begin() + static_cast<std::ptrdiff_t>(steadyFrame(static_cast<std::uint32_t>(times.size()))), times.end());
    if (steady.empty())
        return;
    std::ranges::sort(steady);
    double sum = 0.0;
    for (const double time : steady)
        sum += time;
    std::printf("GPU ms over %zu frames: mean %.2f, median %.2f, p95 %.2f, min %.2f, max %.2f\n", steady.size(), sum / static_cast<double>(steady.size()),
                steady[steady.size() / 2], steady[steady.size() * 95 / 100], steady.front(), steady.back());
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
    std::expected<std::filesystem::path, Error> cache = NoorRay::Core::getCacheDirectory();
    if (!cache)
        return std::unexpected(std::move(cache.error()));
    static const NoorRay::Render::DlssRrExtensions extensions;
    return NoorRay::Core::check(device.open({.enableValidation = std::getenv("NOORRAY_VALIDATION") != nullptr,
                                             .applicationName = "NoorRay",
                                             .extensions = &extensions,
                                             .pipelineDirectory = *cache}));
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

//! A frame records at least once, so that the renderer sees the changes of the frame before it checks for completion.
std::expected<void, Error> renderFrame(NoorRHI::Device& device, NoorRay::Session::Session& session, NoorRHI::Readback<NoorRHI::float4>& readback,
                                       const std::filesystem::path& path, const Options& options) {
    std::uint32_t frames = 0;
    NoorRHI::TimestampQuery query;
    if (options.timings)
        if (auto opened = NoorRay::Core::check(query.open(device)); !opened)
            return opened;
    std::vector<double> times;
    NoorRay::Cli::DlssRrFixture fixture;
    if (options.dlssRrDump)
        if (auto opened = fixture.open(*options.dlssRrDump); !opened)
            return opened;
    NoorRay::Render::GpuProfiler profiler;
    if (options.profile) {
        if (auto opened = profiler.open(device); !opened)
            return opened;
        session.getRenderer().setProfiler(&profiler);
    }
    do {
        NoorRHI::CommandList list(device);
        if (options.timings)
            list.beginTimestamp(query);
        if (auto recorded = session.record(list); !recorded)
            return recorded;
        if (options.timings)
            list.endTimestamp(query);
        if (auto submitted = submitFrame(device, std::move(list)); !submitted)
            return submitted;
        if (options.dlssRrDump)
            if (auto added = fixture.addFrame(session.getRenderer().getDlssRr()); !added)
                return added;
        if (options.timings) {
            times.push_back(query.getMilliseconds());
            if (auto collected = profiler.collect(); !collected)
                return collected;
            if (times.size() == steadyFrame(options.samples))
                profiler.clear();
        }
    } while (rendersRealtime(options) ? ++frames < options.samples : !session.getRenderer().isComplete());
    session.getRenderer().setProfiler(nullptr);
    if (options.dlssRrDump)
        if (auto closed = fixture.close(session.getRenderer().getDlssRr()); !closed)
            return closed;
    if (options.timings)
        printTimings(times, profiler);
    NoorRHI::CommandList list(device);
    NoorRHI::Buffer<NoorRHI::float4> upscaled;
    if (resolvesTemporally(options)) {
        if (auto opened = NoorRay::Core::check(upscaled.open(device, readback.getSize())); !opened)
            return opened;
        if (auto copied = session.getRenderer().recordUpscaledCopy(list, upscaled); !copied)
            return copied;
    }
    list.download(resolvesTemporally(options) ? upscaled : session.getRenderer().getTargets().getOutput(), readback);
    if (auto submitted = submitFrame(device, std::move(list)); !submitted)
        return submitted;
    auto pixels = NoorRay::Core::check(readback.read());
    if (!pixels)
        return std::unexpected(pixels.error());
    if (auto written = writeExr(path, *pixels, options.width, options.height); !written)
        return written;
    std::printf("%s\n", path.string().c_str());
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
    const NoorRay::Format::Animation& animation = session.getAnimation();
    if (options.frames && animation.isEmpty())
        return NoorRay::Core::fail(ErrorCode::InvalidArgument, "--frames needs an animated scene");
    NoorRay::Render::RendererSettings settings = NoorRay::Render::RendererSettings::pathTraced();
    settings.maxSamples = options.samples;
    settings.wavefront.compilesInBackground = false;
    settings.reuse = options.reuse;
    settings.wavefront.gaussianShading = options.gaussianShading;
    settings.denoiser = options.dlssRr    ? NoorRay::Render::Denoiser::OpenDlssRr
                        : options.denoise ? NoorRay::Render::Denoiser::Nrd
                                          : NoorRay::Render::Denoiser::Off;
    //! A realtime run renders samples frames and keeps the last; any other run accumulates samples per pixel at once, without the settle of a viewport.
    settings.accumulate = !rendersRealtime(options);
    settings.convergesWhenStill = false;
    settings.temporal = options.temporal;
    settings.radianceCache = options.cache;
    if (const char* tree = std::getenv("NR_TREE"))
        settings.wavefront.lightTreeDepth = std::atoi(tree);
    settings.wavefront.indirectClamp = options.clamp;
    if (options.depth)
        settings.wavefront.maxDepth = *options.depth;
    if (options.pool)
        settings.wavefront.poolCapacity = *options.pool;
    settings.upscaler = resolvesTemporally(options) && !options.dlssRr ? NoorRay::Render::Upscaler::Fsr : NoorRay::Render::Upscaler::Off;
    settings.renderScale = options.scale.value_or(options.upscale || options.dlssRr ? 0.5f : 1.0f);
    session.getRenderer().setSettings(settings);
    session.setExactVolumeDensity(options.exactVolumes);
    NoorRHI::Readback<NoorRHI::float4> readback;
    if (auto opened = NoorRay::Core::check(readback.open(device, std::size_t{options.width} * options.height)); !opened)
        return opened;
    if (!options.frames)
        return renderFrame(device, session, readback, options.output, options);
    for (int frame = options.frames->first; frame <= options.frames->last; ++frame) {
        if (auto applied = session.setFrame(frame); !applied)
            return applied;
        if (auto rendered = renderFrame(device, session, readback, getFramePath(options.output, frame), options); !rendered)
            return rendered;
    }
    return {};
}

}

int main(int argc, char** argv) {
    NoorRay::Render::prepareDenoiserPlugins();
    auto options = parseOptions(std::span(argv, static_cast<std::size_t>(argc)));
    std::expected<void, Error> result =
        options ? (options->previews ? renderPreviews(*options) : render(*options)) : std::expected<void, Error>(std::unexpected(options.error()));
    if (result)
        return 0;
    std::fprintf(stderr, "%s\n", result.error().message.c_str());
    return 1;
}
