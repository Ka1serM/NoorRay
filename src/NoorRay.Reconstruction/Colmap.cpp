#include "NoorRay.Reconstruction/Colmap.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <deque>
#include <format>
#include <fstream>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include <dlfcn.h>

#include <SDL3/SDL_filesystem.h>

#include "NoorRay.Reconstruction/ColmapBridge/ColmapBridge.h"

namespace NoorRay::Reconstruction {

namespace {

constexpr std::size_t exhaustiveMatchingLimit = 300;
constexpr std::size_t outputTailLines = 20;
constexpr std::string_view undistortedImageSize = "1600";

constexpr std::array counterMarkers{
    std::string_view{"Processed file ["},
    std::string_view{"Processing block ["},
    std::string_view{"Processing image ["},
    std::string_view{"Undistorting image ["},
};

//! Fractions of the global_mapper steps, measured on a 40 image run.
constexpr std::array<std::pair<std::string_view, float>, 5> globalMapperSteps{{
    {"=== Running rotation averaging", 0.02f},
    {"=== Running track establishment", 0.03f},
    {"=== Running global positioning", 0.04f},
    {"=== Running iterative bundle adjustment", 0.14f},
    {"=== Running iterative retriangulation", 0.78f},
}};

struct Counter {
    int index = 0;
    int count = 0;
};

std::optional<Counter> parseCounter(std::string_view& text) {
    Counter counter;
    const char* end = text.data() + text.size();
    const auto [slash, indexError] = std::from_chars(text.data(), end, counter.index);
    if (indexError != std::errc{} || slash == end || *slash != '/')
        return std::nullopt;
    const auto [next, countError] = std::from_chars(slash + 1, end, counter.count);
    if (countError != std::errc{} || counter.count <= 0)
        return std::nullopt;
    text.remove_prefix(static_cast<std::size_t>(next - text.data()));
    return counter;
}

//! COLMAP prints most counters before it processes the item.
std::optional<float> counterFraction(std::string_view text) {
    const std::optional<Counter> outer = parseCounter(text);
    if (!outer)
        return std::nullopt;
    const float outerDone = static_cast<float>(outer->index - 1);
    if (!text.starts_with(", "))
        return outerDone / static_cast<float>(outer->count);
    text.remove_prefix(2);
    const std::optional<Counter> inner = parseCounter(text);
    const float innerDone = inner ? static_cast<float>(inner->index - 1) / static_cast<float>(inner->count) : 0.0f;
    return (outerDone + innerDone) / static_cast<float>(outer->count);
}

std::optional<float> lineFraction(const std::string_view line) {
    for (const std::string_view marker : counterMarkers) {
        const std::size_t at = line.find(marker);
        if (at != std::string_view::npos)
            return counterFraction(line.substr(at + marker.size()));
    }
    for (const auto& [marker, fraction] : globalMapperSteps)
        if (line.contains(marker))
            return fraction;
    return std::nullopt;
}

class ProgressReporter {
public:
    explicit ProgressReporter(const std::function<void(const ColmapProgress&)>& callback) : callback(callback) {}

    void begin(std::string stage, const float start, const float end) {
        progress = {std::move(stage), start};
        stageStart = start;
        stageEnd = end;
        report();
    }

    void line(const std::string_view text) {
        const std::optional<float> fraction = lineFraction(text);
        if (!fraction)
            return;
        const float overall = stageStart + (stageEnd - stageStart) * std::clamp(*fraction, 0.0f, 1.0f);
        if (overall <= progress.fraction)
            return;
        progress.fraction = overall;
        report();
    }

private:
    void report() const {
        if (callback)
            callback(progress);
    }

    const std::function<void(const ColmapProgress&)>& callback;
    ColmapProgress progress;
    float stageStart = 0.0f;
    float stageEnd = 0.0f;
};

class ProcessOutput {
public:
    explicit ProcessOutput(ProgressReporter& reporter) : reporter(reporter) {}

    void append(const std::string_view data) {
        for (const char character : data) {
            if (character != '\n' && character != '\r') {
                pending.push_back(character);
                continue;
            }
            if (!pending.empty())
                finishLine();
        }
    }

    std::string tail() {
        if (!pending.empty())
            finishLine();
        std::string text;
        for (const std::string& line : lines)
            text += std::format("\n{}", line);
        return text;
    }

private:
    void finishLine() {
        reporter.line(pending);
        if (lines.size() == outputTailLines)
            lines.pop_front();
        lines.push_back(std::move(pending));
        pending.clear();
    }

    ProgressReporter& reporter;
    std::string pending;
    std::deque<std::string> lines;
};

constexpr std::string_view bridgeName = "libNoorRayColmap.so";

struct ColmapBridge {
    NoorRayColmapRunFunction run = nullptr;
};

//! Loads the bridge once and keeps it: COLMAP, CUDA and glog do not support an unload.
std::expected<ColmapBridge, Core::Error> loadBridge() {
    static const std::expected<ColmapBridge, Core::Error> bridge = []() -> std::expected<ColmapBridge, Core::Error> {
        const char* base = SDL_GetBasePath();
        const std::string path = std::format("{}{}", base != nullptr ? base : "", bridgeName);
        //! DEEPBIND keeps the libstdc++ of COLMAP and the libc++ of the app apart.
        void* library = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
        if (library == nullptr)
            return Core::fail(Core::ErrorCode::Io, std::format("Cannot load the COLMAP library '{}': {}", path, dlerror()));
        const auto version = reinterpret_cast<NoorRayColmapVersionFunction>(dlsym(library, "noorray_colmap_version"));
        const auto run = reinterpret_cast<NoorRayColmapRunFunction>(dlsym(library, "noorray_colmap_run"));
        if (version == nullptr || run == nullptr || version() != NOORRAY_COLMAP_BRIDGE_VERSION)
            return Core::fail(Core::ErrorCode::Io, std::format("'{}' is not a COLMAP bridge of version {}", path, NOORRAY_COLMAP_BRIDGE_VERSION));
        return ColmapBridge{run};
    }();
    return bridge;
}

std::expected<void, Core::Error> runCommand(const std::vector<std::string>& arguments, const std::stop_token& stop, ProgressReporter& reporter) {
    const std::expected<ColmapBridge, Core::Error> bridge = loadBridge();
    if (!bridge)
        return std::unexpected(bridge.error());
    std::vector<const char*> argv;
    argv.reserve(arguments.size());
    for (const std::string& argument : arguments)
        argv.push_back(argument.c_str());
    struct Context {
        ProcessOutput output;
        const std::stop_token& stop;
    } context{ProcessOutput(reporter), stop};
    const NoorRayColmapCallbacks callbacks{
        .user = &context,
        .line =
            [](void* user, const char* text, const std::size_t size) {
                ProcessOutput& output = static_cast<Context*>(user)->output;
                output.append({text, size});
                output.append("\n");
            },
        .stopped = [](void* user) { return static_cast<Context*>(user)->stop.stop_requested() ? 1 : 0; },
    };
    const NoorRayColmapResult result = bridge->run(static_cast<int>(argv.size()), argv.data(), &callbacks);
    if (result == NoorRayColmapStopped)
        return Core::fail(Core::ErrorCode::InvalidState, std::format("COLMAP {} was stopped", arguments[0]));
    if (result != NoorRayColmapSucceeded)
        return Core::fail(Core::ErrorCode::Io, std::format("COLMAP {} failed:{}", arguments[0], context.output.tail()));
    return {};
}

std::vector<std::string> colmapCommand(const std::string_view command) { return {std::string(command)}; }

bool hasModel(const std::filesystem::path& sparse) {
    std::error_code error;
    return std::filesystem::is_regular_file(sparse / "cameras.bin", error) && std::filesystem::is_regular_file(sparse / "images.bin", error)
           && std::filesystem::is_regular_file(sparse / "points3D.bin", error);
}

std::optional<int> imagesFolderScale(const std::string_view name) {
    if (name == "images")
        return 1;
    if (!name.starts_with("images_"))
        return std::nullopt;
    int scale = 0;
    const char* end = name.data() + name.size();
    const auto [next, error] = std::from_chars(name.data() + 7, end, scale);
    if (error != std::errc{} || next != end || scale <= 0)
        return std::nullopt;
    return scale;
}

std::optional<std::filesystem::path> largestImagesFolder(const std::filesystem::path& dataset) {
    std::optional<std::filesystem::path> best;
    int bestScale = 0;
    std::error_code error;
    for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(dataset, error)) {
        if (!entry.is_directory(error))
            continue;
        const std::optional<int> scale = imagesFolderScale(entry.path().filename().native());
        if (!scale || (best && *scale >= bestScale))
            continue;
        best = entry.path();
        bestScale = *scale;
    }
    return best;
}

bool isPhoto(const std::filesystem::path& path) {
    std::string extension = path.extension().native();
    std::ranges::transform(extension, extension.begin(), [](const unsigned char character) { return static_cast<char>(std::tolower(character)); });
    return extension == ".jpg" || extension == ".jpeg" || extension == ".png";
}

std::vector<std::string> photoNames(const std::filesystem::path& photos) {
    std::vector<std::string> names;
    std::error_code error;
    for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(photos, error))
        if (entry.is_regular_file(error) && isPhoto(entry.path()))
            names.push_back(entry.path().filename().native());
    std::ranges::sort(names);
    return names;
}

std::expected<void, Core::Error> writeImageList(const std::filesystem::path& file, const std::vector<std::string>& names) {
    std::ofstream stream(file);
    for (const std::string& name : names)
        stream << name << '\n';
    if (!stream)
        return Core::fail(Core::ErrorCode::Io, std::format("Cannot write '{}'", file.string()));
    return {};
}

std::uint64_t registeredImages(const std::filesystem::path& model) {
    std::ifstream stream(model / "images.bin", std::ios::binary);
    std::uint64_t count = 0;
    stream.read(reinterpret_cast<char*>(&count), sizeof(count));
    return stream ? count : 0;
}

std::expected<std::filesystem::path, Core::Error> largestModel(const std::filesystem::path& models) {
    std::optional<std::filesystem::path> best;
    std::uint64_t bestCount = 0;
    std::error_code error;
    for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(models, error)) {
        const std::uint64_t count = registeredImages(entry.path());
        if (count <= bestCount)
            continue;
        best = entry.path();
        bestCount = count;
    }
    if (!best)
        return Core::fail(Core::ErrorCode::InvalidState, std::format("COLMAP registered no images in '{}'", models.string()));
    return *best;
}

std::expected<void, Core::Error> moveModelFiles(const std::filesystem::path& sparse, const std::filesystem::path& model) {
    std::error_code error;
    std::filesystem::create_directories(model, error);
    if (error)
        return Core::fail(Core::ErrorCode::Io, std::format("Cannot create '{}': {}", model.string(), error.message()));
    for (const std::string_view name : {"rigs.bin", "frames.bin", "cameras.bin", "images.bin", "points3D.bin"}) {
        std::filesystem::rename(sparse / name, model / name, error);
        if (error)
            return Core::fail(Core::ErrorCode::Io, std::format("Cannot move '{}': {}", (sparse / name).string(), error.message()));
    }
    return {};
}

std::expected<void, Core::Error> resetWorkspace(const std::filesystem::path& workspace) {
    std::error_code error;
    std::filesystem::remove_all(workspace, error);
    if (!error)
        std::filesystem::create_directories(workspace / "mapped", error);
    if (error)
        return Core::fail(Core::ErrorCode::Io, std::format("Cannot prepare '{}': {}", workspace.string(), error.message()));
    return {};
}

std::expected<ColmapDataset, Core::Error> reconstruct(const std::filesystem::path& photos, const std::filesystem::path& workspace, const std::stop_token& stop,
                                                      ProgressReporter& reporter) {
    const std::vector<std::string> names = photoNames(photos);
    if (names.empty())
        return Core::fail(Core::ErrorCode::InvalidArgument, std::format("'{}' has no .jpg, .jpeg or .png photos", photos.string()));
    if (std::expected<void, Core::Error> reset = resetWorkspace(workspace); !reset)
        return std::unexpected(std::move(reset.error()));
    const std::filesystem::path imageList = workspace / "photos.txt";
    if (std::expected<void, Core::Error> written = writeImageList(imageList, names); !written)
        return std::unexpected(std::move(written.error()));
    const std::string database = (workspace / "database.db").string();
    const std::string photoPath = photos.string();

    reporter.begin("Features", 0.0f, 0.2f);
    std::vector<std::string> features = colmapCommand("feature_extractor");
    features.insert(features.end(), {"--database_path", database, "--image_path", photoPath, "--image_list_path", imageList.string(),
                                     "--ImageReader.single_camera", "1", "--ImageReader.camera_model", "OPENCV", "--FeatureExtraction.use_gpu", "1"});
    if (std::expected<void, Core::Error> result = runCommand(features, stop, reporter); !result)
        return std::unexpected(std::move(result.error()));

    reporter.begin("Matching", 0.2f, 0.45f);
    std::vector<std::string> matching = colmapCommand(names.size() <= exhaustiveMatchingLimit ? "exhaustive_matcher" : "sequential_matcher");
    matching.insert(matching.end(), {"--database_path", database, "--FeatureMatching.use_gpu", "1"});
    if (std::expected<void, Core::Error> result = runCommand(matching, stop, reporter); !result)
        return std::unexpected(std::move(result.error()));

    reporter.begin("Mapping", 0.45f, 0.9f);
    const std::filesystem::path models = workspace / "mapped";
    std::vector<std::string> mapping = colmapCommand("global_mapper");
    mapping.insert(mapping.end(), {"--database_path", database, "--image_path", photoPath, "--output_path", models.string()});
    if (std::expected<void, Core::Error> result = runCommand(mapping, stop, reporter); !result)
        return std::unexpected(std::move(result.error()));
    const std::expected<std::filesystem::path, Core::Error> model = largestModel(models);
    if (!model)
        return std::unexpected(model.error());

    reporter.begin("Undistortion", 0.9f, 1.0f);
    std::vector<std::string> undistortion = colmapCommand("image_undistorter");
    undistortion.insert(undistortion.end(), {"--image_path", photoPath, "--input_path", model->string(), "--output_path", workspace.string(), "--output_type",
                                             "COLMAP", "--max_image_size", std::string(undistortedImageSize)});
    if (std::expected<void, Core::Error> result = runCommand(undistortion, stop, reporter); !result)
        return std::unexpected(std::move(result.error()));
    if (std::expected<void, Core::Error> moved = moveModelFiles(workspace / "sparse", workspace / "sparse" / "0"); !moved)
        return std::unexpected(std::move(moved.error()));
    return ColmapDataset{workspace / "images", workspace / "sparse" / "0"};
}

std::optional<ColmapDataset> existingDataset(const std::filesystem::path& folder) {
    if (!hasModel(folder / "sparse" / "0"))
        return std::nullopt;
    const std::optional<std::filesystem::path> images = largestImagesFolder(folder);
    if (!images)
        return std::nullopt;
    return ColmapDataset{*images, folder / "sparse" / "0"};
}

}

std::expected<ColmapDataset, Core::Error> runColmap(const std::filesystem::path& photos, const std::stop_token stop,
                                                    const std::function<void(const ColmapProgress&)>& progress) {
    if (std::optional<ColmapDataset> dataset = existingDataset(photos))
        return *std::move(dataset);
    const std::filesystem::path workspace = photos / "colmap";
    if (std::optional<ColmapDataset> dataset = existingDataset(workspace))
        return *std::move(dataset);
    ProgressReporter reporter(progress);
    return reconstruct(photos, workspace, stop, reporter);
}

}
