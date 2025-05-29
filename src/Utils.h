#pragma once

#include <vector>
#include <string>

#include "Shaders/SharedStructs.h"
#include "Vulkan/Renderer.h"

class Utils {
public:
    static void loadObj(Context& context, Renderer& renderer, const std::string& filepath, std::vector<Vertex>& vertices, std::vector<uint32_t>
                        & indices, std::vector<Face>& faces);

    static std::vector<char> readFile(const std::string& filename);
};
