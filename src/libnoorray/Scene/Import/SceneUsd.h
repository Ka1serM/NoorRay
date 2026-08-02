#pragma once

#include <string>

class Scene;

namespace nr::sceneio {

bool isUsdFile(const std::string& filepath);
void readUsd(Scene& scene, const std::string& filepath);
void writeUsd(const Scene& scene, const std::string& filepath);

} // namespace nr::sceneio
