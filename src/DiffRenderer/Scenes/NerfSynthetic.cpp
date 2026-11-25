#include <charconv>
#include <cmath>
#include <format>
#include <fstream>
#include <sstream>

#include "DiffRenderer/Scenes/Loading.h"

namespace DiffRenderer::Scenes {

namespace {

constexpr std::uint32_t imageSize = 800;
constexpr std::uint32_t startPointCount = 100'000;
constexpr float startPointBound = 1.3f;
constexpr std::uint64_t startPointSeed = 1;

class JsonScanner {
public:
    explicit JsonScanner(std::string text) : text(std::move(text)) {}

    bool seekKey(const std::string_view key) {
        const std::size_t found = text.find(std::format("\"{}\"", key), position);
        if (found == std::string::npos)
            return false;
        position = text.find(':', found) + 1;
        return true;
    }

    std::optional<double> readNumber() {
        position = text.find_first_of("-0123456789", position);
        if (position == std::string::npos)
            return std::nullopt;
        double value = 0.0;
        const auto [end, error] = std::from_chars(text.data() + position, text.data() + text.size(), value);
        if (error != std::errc())
            return std::nullopt;
        position = static_cast<std::size_t>(end - text.data());
        return value;
    }

    std::optional<std::string> readString() {
        const std::size_t open = text.find('"', position);
        const std::size_t close = open == std::string::npos ? open : text.find('"', open + 1);
        if (close == std::string::npos)
            return std::nullopt;
        position = close + 1;
        return text.substr(open + 1, close - open - 1);
    }

private:
    std::string text;
    std::size_t position = 0;
};

Camera makeNerfCamera(const std::array<double, 16>& cameraToWorld, const double focal) {
    Matrix3 rotation;
    Vector3 position{cameraToWorld[3], cameraToWorld[7], cameraToWorld[11]};
    for (std::size_t row = 0; row < 3; ++row)
        for (std::size_t column = 0; column < 3; ++column)
            rotation[column][row] = cameraToWorld[row * 4 + column] * (column == 0 ? 1.0 : -1.0);
    Vector3 translation{};
    for (std::size_t row = 0; row < 3; ++row)
        translation[row] = -(rotation[row][0] * position[0] + rotation[row][1] * position[1] + rotation[row][2] * position[2]);
    const auto f = static_cast<float>(focal);
    constexpr float center = 0.5f * imageSize;
    return makeCamera(rotation, translation, {f, f, center, center}, imageSize, imageSize);
}

std::expected<std::vector<ViewSource>, std::string> readTransforms(const std::filesystem::path& folder, const std::string_view split) {
    const std::filesystem::path path = folder / std::format("transforms_{}.json", split);
    std::ifstream stream(path);
    if (!stream)
        return std::unexpected(std::format("{} does not open", path.string()));
    std::stringstream text;
    text << stream.rdbuf();
    JsonScanner scanner(text.str());
    const std::optional<double> angle = scanner.seekKey("camera_angle_x") ? scanner.readNumber() : std::nullopt;
    if (!angle)
        return std::unexpected(std::format("{} has no camera_angle_x", path.string()));
    const double focal = 0.5 * imageSize / std::tan(0.5 * *angle);
    std::vector<ViewSource> views;
    while (scanner.seekKey("file_path")) {
        const std::optional<std::string> file = scanner.readString();
        if (!file || !scanner.seekKey("transform_matrix"))
            return std::unexpected(std::format("{} has a frame without file_path or transform_matrix", path.string()));
        std::array<double, 16> matrix{};
        for (double& value : matrix) {
            const std::optional<double> number = scanner.readNumber();
            if (!number)
                return std::unexpected(std::format("{} has an incomplete transform_matrix", path.string()));
            value = *number;
        }
        const std::filesystem::path relative(*file + ".png");
        views.push_back({relative.stem().string(), makeNerfCamera(matrix, focal), folder / relative});
    }
    return views;
}

}

std::expected<Dataset, std::string> loadNerfSynthetic(const std::string_view object, const std::filesystem::path& folder, const LoadOptions& options) {
    DatasetSource source;
    std::expected<std::vector<ViewSource>, std::string> train = readTransforms(folder, "train");
    if (!train)
        return std::unexpected(std::move(train.error()));
    std::expected<std::vector<ViewSource>, std::string> test = readTransforms(folder, "test");
    if (!test)
        return std::unexpected(std::move(test.error()));
    if (train->empty() || test->empty())
        return std::unexpected(std::format("{} has no train or test views", object));
    source.train = std::move(*train);
    source.test = std::move(*test);
    Dataset dataset;
    dataset.background = {1, 1, 1};
    dataset.points = makeRandomPoints(startPointCount, -startPointBound, startPointBound, startPointSeed);
    setBounds(dataset, source.train);
    if (std::expected<void, std::string> loaded = loadViews(dataset, source, options); !loaded)
        return std::unexpected(std::move(loaded.error()));
    return dataset;
}

}
