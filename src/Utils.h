#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include "shaders/SharedStructs.h"

class Utils {
public:
    static void loadObj(std::string filename, std::vector<Vertex> &vertices, std::vector<uint32_t> &indices, std::vector<Face> &faces, std::
                        vector<Material> &materials, std::vector<PointLight> &pointLights);
    static std::vector<char> readFile(const std::string& filename);
};
