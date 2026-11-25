#include "NoorRay.Reconstruction/Datasets.h"

#include <algorithm>
#include <cctype>
#include <tuple>

namespace NoorRay::Reconstruction {

namespace {

bool isPhoto(const std::filesystem::path& path) {
    std::string extension = path.extension().native();
    std::ranges::transform(extension, extension.begin(), [](const unsigned char character) { return static_cast<char>(std::tolower(character)); });
    return extension == ".jpg" || extension == ".jpeg" || extension == ".png";
}

bool hasPhotos(const std::filesystem::path& folder) {
    std::error_code error;
    for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(folder, error))
        if (entry.is_regular_file(error) && isPhoto(entry.path()))
            return true;
    return false;
}

std::vector<std::filesystem::path> subfolders(const std::filesystem::path& folder) {
    std::vector<std::filesystem::path> folders;
    std::error_code error;
    for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(folder, error))
        if (entry.is_directory(error) && !entry.path().filename().native().starts_with('.'))
            folders.push_back(entry.path());
    return folders;
}

}

std::optional<DatasetKind> detectDataset(const std::filesystem::path& folder) {
    std::error_code error;
    if (std::filesystem::is_regular_file(folder / "transforms_train.json", error))
        return DatasetKind::Synthetic;
    if (std::filesystem::is_regular_file(folder / "sparse" / "0" / "images.bin", error))
        return DatasetKind::Posed;
    if (hasPhotos(folder))
        return DatasetKind::Photos;
    return std::nullopt;
}

std::vector<DatasetEntry> scanDatasets(const std::filesystem::path& root) {
    std::vector<DatasetEntry> entries;
    for (const std::filesystem::path& folder : subfolders(root)) {
        if (const std::optional<DatasetKind> kind = detectDataset(folder)) {
            entries.push_back({"", folder.filename().string(), *kind, folder});
            continue;
        }
        for (const std::filesystem::path& scene : subfolders(folder))
            if (const std::optional<DatasetKind> kind = detectDataset(scene))
                entries.push_back({folder.filename().string(), scene.filename().string(), *kind, scene});
    }
    std::ranges::sort(entries, {}, [](const DatasetEntry& entry) { return std::tie(entry.group, entry.name); });
    return entries;
}

const char* getDatasetKindName(const DatasetKind kind) {
    switch (kind) {
    case DatasetKind::Posed:
        return "COLMAP poses";
    case DatasetKind::Synthetic:
        return "synthetic, known poses";
    case DatasetKind::Photos:
        return "photographs, COLMAP runs first";
    }
    return "";
}

}
