#pragma once

#include <string>

class Scene;

class SceneReader {
public:
    static void Read(Scene& scene, const std::string& filepath);
};
