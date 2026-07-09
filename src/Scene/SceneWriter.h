#pragma once

#include <string>

class Scene;

class SceneWriter {
public:
    static void Write(const Scene& scene, const std::string& filepath);
};
