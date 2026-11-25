#include "ColmapBridge.h"

#include <chrono>
#include <exception>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <colmap/controllers/feature_extraction.h>
#include <colmap/controllers/feature_matching.h>
#include <colmap/controllers/option_manager.h>
#include <colmap/controllers/undistorters.h>
#include <colmap/exe/feature.h>
#include <colmap/exe/sfm.h>
#include <colmap/feature/sift.h>
#include <colmap/scene/reconstruction.h>
#include <colmap/scene/reconstruction_manager.h>
#include <colmap/sensor/models.h>
#include <colmap/util/file.h>
#include <colmap/util/string.h>
#include <colmap/util/threading.h>
#include <glog/logging.h>

namespace {

constexpr std::chrono::milliseconds stopPollInterval{50};

//! Sends the glog messages of COLMAP to the plugin instead of to files or stderr.
class LineSink final : public google::LogSink {
public:
    explicit LineSink(const NoorRayColmapCallbacks& callbacks) : callbacks(callbacks) { google::AddLogSink(this); }
    ~LineSink() override { google::RemoveLogSink(this); }
    LineSink(const LineSink&) = delete;
    LineSink& operator=(const LineSink&) = delete;

    void send(google::LogSeverity, const char*, const char*, int, const google::LogMessageTime&, const char* message, const size_t size) override {
        write({message, size});
    }

    void write(const std::string_view text) {
        if (callbacks.line == nullptr)
            return;
        const std::lock_guard lock(mutex);
        callbacks.line(callbacks.user, text.data(), text.size());
    }

private:
    const NoorRayColmapCallbacks& callbacks;
    std::mutex mutex;
};

bool isStopped(const NoorRayColmapCallbacks& callbacks) { return callbacks.stopped != nullptr && callbacks.stopped(callbacks.user) != 0; }

void initializeLogging() {
    static std::once_flag once;
    std::call_once(once, [] {
        google::InitGoogleLogging("noorray-colmap");
        //! An empty base name turns the log files of a severity off.
        for (const google::LogSeverity severity : {google::GLOG_INFO, google::GLOG_WARNING, google::GLOG_ERROR, google::GLOG_FATAL})
            google::SetLogDestination(severity, "");
        FLAGS_stderrthreshold = google::GLOG_FATAL;
    });
}

//! Runs a COLMAP thread and stops it when the plugin asks.
NoorRayColmapResult runThread(colmap::Thread& thread, const NoorRayColmapCallbacks& callbacks) {
    thread.Start();
    bool stopped = false;
    while (thread.IsRunning()) {
        if (!stopped && isStopped(callbacks)) {
            thread.Stop();
            stopped = true;
        }
        std::this_thread::sleep_for(stopPollInterval);
    }
    thread.Wait();
    return stopped ? NoorRayColmapStopped : NoorRayColmapSucceeded;
}

NoorRayColmapResult runController(colmap::BaseController& controller, const NoorRayColmapCallbacks& callbacks) {
    controller.SetCheckIfStoppedFunc([&callbacks] { return isStopped(callbacks); });
    controller.Run();
    return isStopped(callbacks) ? NoorRayColmapStopped : NoorRayColmapSucceeded;
}

NoorRayColmapResult runFeatureExtractor(int argc, char** argv, const NoorRayColmapCallbacks& callbacks) {
    std::filesystem::path imageListPath;
    int cameraMode = -1;
    std::string descriptorNormalization = "l1_root";
    colmap::OptionManager options;
    options.AddDatabaseOptions();
    options.AddImageOptions();
    options.AddDefaultOption("camera_mode", &cameraMode);
    options.AddDefaultOption("image_list_path", &imageListPath);
    options.AddDefaultOption("descriptor_normalization", &descriptorNormalization, "{'l1_root', 'l2'}");
    options.AddFeatureExtractionOptions();
    if (!options.Parse(argc, argv))
        return NoorRayColmapFailed;
    colmap::ImageReaderOptions reader = *options.image_reader;
    reader.image_path = *options.image_path;
    reader.as_rgb = options.feature_extraction->RequiresRGB();
    if (cameraMode >= 0)
        colmap::UpdateImageReaderOptionsFromCameraMode(reader, static_cast<colmap::CameraMode>(cameraMode));
    colmap::StringToUpper(&descriptorNormalization);
    options.feature_extraction->sift->normalization = colmap::SiftExtractionOptions::NormalizationFromString(descriptorNormalization);
    if (!imageListPath.empty()) {
        reader.image_names = colmap::ReadTextFileLines(imageListPath);
        if (reader.image_names.empty())
            return NoorRayColmapSucceeded;
    }
    if (!colmap::ExistsCameraModelWithName(reader.camera_model) || !colmap::VerifyCameraParams(reader.camera_model, reader.camera_params)) {
        LOG(ERROR) << "Invalid camera model or camera parameters";
        return NoorRayColmapFailed;
    }
    //! The GUI-less build extracts with CUDA, so no OpenGL context is needed.
    if (options.feature_extraction->RequiresOpenGL()) {
        LOG(ERROR) << "This feature extraction needs OpenGL, which the bridge does not create";
        return NoorRayColmapFailed;
    }
    const std::unique_ptr<colmap::Thread> extractor = colmap::CreateFeatureExtractorController(*options.database_path, reader, *options.feature_extraction);
    return runThread(*extractor, callbacks);
}

NoorRayColmapResult runMatcher(const std::string_view command, int argc, char** argv, const NoorRayColmapCallbacks& callbacks) {
    colmap::OptionManager options;
    options.AddDatabaseOptions();
    const bool exhaustive = command == "exhaustive_matcher";
    if (exhaustive)
        options.AddExhaustivePairingOptions();
    else
        options.AddSequentialPairingOptions();
    if (!options.Parse(argc, argv))
        return NoorRayColmapFailed;
    if (options.feature_matching->RequiresOpenGL()) {
        LOG(ERROR) << "This feature matching needs OpenGL, which the bridge does not create";
        return NoorRayColmapFailed;
    }
    const std::unique_ptr<colmap::Thread> matcher =
        exhaustive ? colmap::CreateExhaustiveFeatureMatcher(*options.exhaustive_pairing, *options.feature_matching, *options.two_view_geometry,
                                                            *options.database_path)
                   : colmap::CreateSequentialFeatureMatcher(*options.sequential_pairing, *options.feature_matching, *options.two_view_geometry,
                                                            *options.database_path);
    return runThread(*matcher, callbacks);
}

NoorRayColmapResult runGlobalMapper(int argc, char** argv, const NoorRayColmapCallbacks& callbacks) {
    std::filesystem::path outputPath;
    colmap::OptionManager options;
    options.AddDatabaseOptions();
    options.AddImageOptions();
    options.AddRequiredOption("output_path", &outputPath);
    options.AddGlobalMapperOptions();
    if (!options.Parse(argc, argv))
        return NoorRayColmapFailed;
    if (!colmap::ExistsDir(outputPath)) {
        LOG(ERROR) << "`output_path` is not a directory.";
        return NoorRayColmapFailed;
    }
    if (isStopped(callbacks))
        return NoorRayColmapStopped;
    auto reconstructions = std::make_shared<colmap::ReconstructionManager>();
    if (!colmap::RunGlobalMapperImpl(*options.database_path, *options.image_path, outputPath, options.global_mapper, reconstructions))
        return NoorRayColmapFailed;
    return isStopped(callbacks) ? NoorRayColmapStopped : NoorRayColmapSucceeded;
}

NoorRayColmapResult runImageUndistorter(int argc, char** argv, const NoorRayColmapCallbacks& callbacks) {
    std::filesystem::path inputPath;
    std::filesystem::path outputPath;
    std::string outputType = "COLMAP";
    colmap::COLMAPUndistorter::Options undistorter;
    colmap::UndistortCameraOptions camera;
    colmap::OptionManager options;
    options.AddImageOptions();
    options.AddRequiredOption("input_path", &inputPath);
    options.AddRequiredOption("output_path", &outputPath);
    options.AddDefaultOption("output_type", &outputType, "{COLMAP}");
    options.AddDefaultOption("max_image_size", &camera.max_image_size);
    options.AddDefaultOption("jpeg_quality", &undistorter.jpeg_quality);
    options.AddDefaultOption("num_threads", &undistorter.num_threads);
    if (!options.Parse(argc, argv))
        return NoorRayColmapFailed;
    if (outputType != "COLMAP") {
        LOG(ERROR) << "The bridge only writes the COLMAP output type";
        return NoorRayColmapFailed;
    }
    colmap::CreateDirIfNotExists(outputPath);
    colmap::Reconstruction reconstruction;
    reconstruction.Read(inputPath);
    colmap::COLMAPUndistorter controller(undistorter, camera, reconstruction, *options.image_path, outputPath);
    return runController(controller, callbacks);
}

NoorRayColmapResult runCommand(const std::string_view command, int argc, char** argv, const NoorRayColmapCallbacks& callbacks) {
    if (command == "feature_extractor")
        return runFeatureExtractor(argc, argv, callbacks);
    if (command == "exhaustive_matcher" || command == "sequential_matcher")
        return runMatcher(command, argc, argv, callbacks);
    if (command == "global_mapper")
        return runGlobalMapper(argc, argv, callbacks);
    if (command == "image_undistorter")
        return runImageUndistorter(argc, argv, callbacks);
    LOG(ERROR) << "Unknown COLMAP command " << command;
    return NoorRayColmapFailed;
}

}

extern "C" int noorray_colmap_version(void) { return NOORRAY_COLMAP_BRIDGE_VERSION; }

extern "C" NoorRayColmapResult noorray_colmap_run(const int argumentCount, const char* const* arguments, const NoorRayColmapCallbacks* callbacks) {
    static const NoorRayColmapCallbacks none{};
    const NoorRayColmapCallbacks& used = callbacks != nullptr ? *callbacks : none;
    if (argumentCount < 1 || arguments == nullptr)
        return NoorRayColmapFailed;
    initializeLogging();
    LineSink sink(used);
    try {
        //! OptionManager parses like the colmap tool: argv[0] is the program, then the options; it logs to the sink instead of to files.
        std::vector<std::string> storage{"colmap"};
        for (int index = 1; index < argumentCount; ++index)
            storage.emplace_back(arguments[index]);
        storage.insert(storage.end(), {"--log_target", "file"});
        std::vector<char*> argv;
        for (std::string& argument : storage)
            argv.push_back(argument.data());
        argv.push_back(nullptr);
        return runCommand(arguments[0], static_cast<int>(storage.size()), argv.data(), used);
    } catch (const std::exception& exception) {
        sink.write(std::string("COLMAP failed: ") + exception.what());
    } catch (...) {
        sink.write("COLMAP failed with an unknown exception");
    }
    return NoorRayColmapFailed;
}
