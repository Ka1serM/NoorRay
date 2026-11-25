#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace NoorRay::Reconstruction {

enum class DatasetKind : std::uint8_t {
    //! A COLMAP model in sparse/0 and the images in images or images_N, as Mip-NeRF 360.
    Posed,
    //! transforms_train.json and transforms_test.json with RGBA renders, as NeRF synthetic.
    Synthetic,
    //! Only photographs: COLMAP finds the poses first.
    Photos,
};

struct DatasetEntry {
    std::string group;
    std::string name;
    DatasetKind kind = DatasetKind::Photos;
    std::filesystem::path folder;
};

std::optional<DatasetKind> detectDataset(const std::filesystem::path& folder);
//! The datasets directly in the root and in its subfolders, sorted by group and name.
std::vector<DatasetEntry> scanDatasets(const std::filesystem::path& root);
const char* getDatasetKindName(DatasetKind kind);

}
