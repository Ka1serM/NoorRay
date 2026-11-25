#pragma once

#include <memory>
#include <tuple>
#include <vector>

#include "NoorRay/Materials/Graph.h"
#include "NoorRay/Materials/NodeTree.h"
#include "NoorRay/Materials/Program.h"
#include "NoorRay/Scene/Scene.h"

namespace NoorRay::Format {

//! What the indices of a scene file name: the scene objects of the file and the objects that they share, in file order.
class SceneIndex {
public:
    template <class Object>
    std::vector<Core::Handle<Object>>& get() {
        return std::get<std::vector<Core::Handle<Object>>>(handles);
    }
    template <class Object>
    const std::vector<Core::Handle<Object>>& get() const {
        return std::get<std::vector<Core::Handle<Object>>>(handles);
    }
    template <class Object>
    std::vector<std::shared_ptr<const Object>>& getShared() {
        return std::get<std::vector<std::shared_ptr<const Object>>>(shared);
    }
    template <class Object>
    const std::vector<std::shared_ptr<const Object>>& getShared() const {
        return std::get<std::vector<std::shared_ptr<const Object>>>(shared);
    }

private:
    std::tuple<std::vector<Scene::TextureHandle>, std::vector<Scene::MaterialHandle>, std::vector<Scene::MeshHandle>, std::vector<Scene::InstanceHandle>,
               std::vector<Scene::LightHandle>, std::vector<Scene::GaussianCloudHandle>, std::vector<Scene::VolumeHandle>, std::vector<Scene::CameraHandle>>
        handles;
    std::tuple<std::vector<std::shared_ptr<const Materials::Graph>>, std::vector<std::shared_ptr<const Materials::Program>>,
               std::vector<std::shared_ptr<const Materials::NodeTree>>>
        shared;
};

}
