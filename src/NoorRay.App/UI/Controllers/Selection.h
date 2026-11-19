#pragma once

#include <variant>

#include "NoorRay/Scene/Scene.h"

namespace NoorRay::App {

using Selection = std::variant<std::monostate, Scene::InstanceHandle, Scene::LightHandle, Scene::CameraHandle, Scene::GaussianCloudHandle, Scene::VolumeHandle>;

}
