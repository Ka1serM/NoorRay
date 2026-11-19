#pragma once

#include <string>
#include <variant>
#include <vector>

#include "NoorRay/Core/Handle.h"
#include "NoorRay/Core/Math.h"

namespace NoorRay::Scene {

struct Instance;
struct Light;
struct GaussianCloud;
struct Volume;
struct Camera;

using NodeObject = std::variant<std::monostate, Core::Handle<Instance>, Core::Handle<Light>, Core::Handle<GaussianCloud>, Core::Handle<Volume>, Core::Handle<Camera>>;

//! A node without an object is a group. The scene keeps the world transform and the children, and writes the world transform into the object.
struct Node {
    std::string name;
    Core::Transform localTransform;
    Core::Handle<Node> parent;
    NodeObject object;
    Core::Transform worldTransform;
    std::vector<Core::Handle<Node>> children;
};

}
