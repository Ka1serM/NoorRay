#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include "Mesh.h"

class Utils {
public:
    static void loadFromFile(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices, std::vector<Face>& faces);
    static std::vector<char> readFile(const std::string& filename);
};