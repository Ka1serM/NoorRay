#include "Training/ColmapReconstruction.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#ifndef _WIN32
#include <unistd.h>
#endif

#include <glm/gtc/quaternion.hpp>

namespace noorray
{
namespace
{
struct Camera
{
    uint32_t width{};
    uint32_t height{};
    double fx{}, fy{}, cx{}, cy{};
};

std::string shellQuote(const std::string& value)
{
#ifdef _WIN32
    std::string quoted = "\"";
    for (const char c : value) quoted += c == '"' ? "\\\"" : std::string(1, c);
    return quoted + '"';
#else
    std::string quoted = "'";
    for (const char c : value) quoted += c == '\'' ? "'\\''" : std::string(1, c);
    return quoted + '\'';
#endif
}

std::string colmapExecutable()
{
    const char* pathValue = std::getenv("PATH");
    if (!pathValue)
        throw std::runtime_error("COLMAP was not found because PATH is not set");
#ifdef _WIN32
    constexpr char separator = ';';
    constexpr const char* executableName = "colmap.exe";
#else
    constexpr char separator = ':';
    constexpr const char* executableName = "colmap";
#endif
    std::istringstream paths(pathValue);
    std::string directory;
    while (std::getline(paths, directory, separator))
    {
        if (directory.empty()) directory = ".";
        const std::filesystem::path candidate = std::filesystem::path(directory) / executableName;
#ifdef _WIN32
        if (std::filesystem::is_regular_file(candidate)) return candidate.string();
#else
        if (std::filesystem::is_regular_file(candidate) && ::access(candidate.c_str(), X_OK) == 0)
            return candidate.string();
#endif
    }
    throw std::runtime_error("COLMAP was not found in PATH. Install its prebuilt binary first");
}

void runColmap(const std::string& command, const std::vector<std::string>& arguments)
{
    std::string invocation = shellQuote(colmapExecutable()) + " " + command;
    for (const std::string& argument : arguments)
        invocation += " " + shellQuote(argument);
    if (std::system(invocation.c_str()) != EXIT_SUCCESS)
        throw std::runtime_error("COLMAP " + command + " failed");
}

std::unordered_map<uint64_t, Camera> readCameras(const std::filesystem::path& path)
{
    std::ifstream file(path);
    if (!file) throw std::runtime_error("COLMAP did not produce cameras.txt");
    std::unordered_map<uint64_t, Camera> cameras;
    std::string line;
    while (std::getline(file, line))
    {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream stream(line);
        uint64_t id;
        std::string model;
        Camera camera;
        stream >> id >> model >> camera.width >> camera.height;
        if (model == "PINHOLE")
        {
            stream >> camera.fx >> camera.fy >> camera.cx >> camera.cy;
        }
        else if (model == "SIMPLE_PINHOLE")
        {
            stream >> camera.fx >> camera.cx >> camera.cy;
            camera.fy = camera.fx;
        }
        else
            throw std::runtime_error("Training requires COLMAP-undistorted PINHOLE cameras, got " + model);
        cameras.emplace(id, camera);
    }
    return cameras;
}

glm::mat4 cameraToWorld(double qw, double qx, double qy, double qz, const glm::dvec3& translation)
{
    const glm::dmat3 worldToCamera = glm::mat3_cast(glm::normalize(glm::dquat(qw, qx, qy, qz)));
    const glm::dmat3 rotation = glm::transpose(worldToCamera);
    const glm::dvec3 position = -rotation * translation;
    glm::mat4 result(1.0f);
    for (int column = 0; column < 3; ++column)
        for (int row = 0; row < 3; ++row)
            result[column][row] = static_cast<float>(rotation[column][row]);
    result[3] = glm::vec4(position, 1.0);
    result[1] *= -1.0f;
    result[2] *= -1.0f;
    return result;
}

void readImages(const std::filesystem::path& path, const std::filesystem::path& imageDirectory,
    const std::unordered_map<uint64_t, Camera>& cameras, ColmapReconstruction& result)
{
    std::ifstream file(path);
    if (!file) throw std::runtime_error("COLMAP did not produce images.txt");
    std::string line;
    while (std::getline(file, line))
    {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream stream(line);
        uint64_t imageId, cameraId;
        double qw, qx, qy, qz, tx, ty, tz;
        stream >> imageId >> qw >> qx >> qy >> qz >> tx >> ty >> tz >> cameraId;
        std::string name;
        std::getline(stream >> std::ws, name);
        const Camera& camera = cameras.at(cameraId);
        result.views.push_back({imageDirectory / name,
            cameraToWorld(qw, qx, qy, qz, {tx, ty, tz}),
            static_cast<float>(camera.fx), static_cast<float>(camera.fy),
            static_cast<float>(camera.cx), static_cast<float>(camera.cy),
            camera.width, camera.height});
        std::getline(file, line); // POINTS2D line
    }
}

void readPoints(const std::filesystem::path& path, ColmapReconstruction& result)
{
    std::ifstream file(path);
    if (!file) throw std::runtime_error("COLMAP did not produce points3D.txt");
    std::string line;
    while (std::getline(file, line))
    {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream stream(line);
        uint64_t id;
        double x, y, z, error;
        int r, g, b;
        stream >> id >> x >> y >> z >> r >> g >> b >> error;
        result.points.emplace_back(x, y, z);
        result.colors.emplace_back(r / 255.0f, g / 255.0f, b / 255.0f);
    }
}

std::filesystem::path writeImageList(const std::filesystem::path& imageDirectory,
    const std::filesystem::path& workspaceDirectory)
{
    const std::filesystem::path path = workspaceDirectory / "images.txt";
    std::ofstream file(path);
    if (!file) throw std::runtime_error("Could not create the COLMAP image list");
    size_t count = 0;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(imageDirectory))
    {
        if (!entry.is_regular_file()) continue;
        std::string extension = entry.path().extension().string();
        std::ranges::transform(extension, extension.begin(),
            [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (extension != ".jpg" && extension != ".jpeg" && extension != ".png" &&
            extension != ".bmp" && extension != ".tif" && extension != ".tiff")
            continue;
        file << std::filesystem::relative(entry.path(), imageDirectory).generic_string() << '\n';
        ++count;
    }
    if (count == 0) throw std::runtime_error("The selected folder contains no supported images");
    return path;
}
}

ColmapReconstruction reconstructImagesWithColmap(
    const std::filesystem::path& imageDirectory,
    const std::filesystem::path& workspaceDirectory,
    const ReconstructionMapper mapper)
{
    if (!std::filesystem::is_directory(imageDirectory))
        throw std::runtime_error("The selected image directory does not exist");
    std::filesystem::create_directories(workspaceDirectory);
    const auto database = workspaceDirectory / "database.db";
    const auto sparse = workspaceDirectory / "sparse";
    const auto textModel = workspaceDirectory / "sparse_text";
    const auto undistorted = workspaceDirectory / "undistorted";
    std::filesystem::remove(database);
    std::filesystem::remove_all(sparse);
    std::filesystem::remove_all(textModel);
    std::filesystem::remove_all(undistorted);
    std::filesystem::create_directories(sparse);
    std::filesystem::create_directories(textModel);
    const auto imageList = writeImageList(imageDirectory, workspaceDirectory);

    runColmap("feature_extractor", {"--database_path", database.string(), "--image_path",
        imageDirectory.string(), "--image_list_path", imageList.string(), "--ImageReader.single_camera", "1",
        "--FeatureExtraction.use_gpu", "1"});
    runColmap("exhaustive_matcher", {"--database_path", database.string(),
        "--FeatureMatching.use_gpu", "1"});
    const std::string mapperCommand = mapper == ReconstructionMapper::Global
        ? "global_mapper" : "mapper";
    runColmap(mapperCommand, {"--database_path", database.string(), "--image_path",
        imageDirectory.string(), "--output_path", sparse.string()});

    const auto model = sparse / "0";
    if (!std::filesystem::is_directory(model))
        throw std::runtime_error("COLMAP could not register an image reconstruction");
    runColmap("image_undistorter", {"--image_path", imageDirectory.string(),
        "--input_path", model.string(), "--output_path", undistorted.string(),
        "--output_type", "COLMAP", "--max_image_size", "1024"});
    const auto undistortedModel = undistorted / "sparse";
    runColmap("model_converter", {"--input_path", undistortedModel.string(), "--output_path",
        textModel.string(), "--output_type", "TXT"});

    ColmapReconstruction result;
    const auto cameras = readCameras(textModel / "cameras.txt");
    readPoints(textModel / "points3D.txt", result);
    readImages(textModel / "images.txt", undistorted / "images", cameras, result);
    return result;
}
}
