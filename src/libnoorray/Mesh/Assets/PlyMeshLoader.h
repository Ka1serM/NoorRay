#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

#include "Mesh/Assets/MeshAsset.h"

struct PlyMeshData {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
};

class PlyMeshLoader {
public:
    static bool HasFaces(const std::filesystem::path& path);
    static PlyMeshData Load(const std::filesystem::path& path);
};
