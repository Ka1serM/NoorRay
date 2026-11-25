#pragma once

#include <variant>

#include "NoorRay/Scene/Scene.h"

namespace NoorRay::App {

//! A node handle selects an empty: a node without an object, which has only its transform.
using Selection = std::variant<std::monostate, Scene::InstanceHandle, Scene::LightHandle, Scene::CameraHandle, Scene::GaussianCloudHandle, Scene::VolumeHandle,
                               Scene::NodeHandle>;

}
