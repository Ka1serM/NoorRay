#include "Bindings.h"

#include <memory>

#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

#include "Camera/CameraInstance.h"
#include "Mesh/MeshAsset.h"
#include "Scene/Scene.h"
#include "Scene/SceneObject.h"

namespace nb = nanobind;
using namespace nb::literals;

void bindScene(nb::module_& module)
{
    nb::class_<Scene>(module, "Scene")
        .def("load", &Scene::load, "path"_a)
        .def("import_file", &Scene::importFile, "path"_a)
        .def("read", &Scene::read, "path"_a)
        .def("clear", &Scene::clear)
        .def("add", nb::overload_cast<std::unique_ptr<SceneObject>>(&Scene::add), "object"_a)
        .def("add_camera", [](Scene& scene, std::unique_ptr<Camera> camera,
                              const std::string& name, Transform transform) {
            nb::object pythonCamera = nb::find(camera.get());
            auto instance = std::make_unique<CameraInstance>(
                std::move(camera), name, std::move(transform));
            const uint64_t id = scene.add(std::move(instance));
            if (pythonCamera.is_valid())
                nb::inst_set_state(pythonCamera, true, false);
            return std::dynamic_pointer_cast<CameraInstance>(scene.getObjectPtr(id));
        }, "camera"_a, "name"_a = "Camera", "transform"_a = Transform{},
           nb::rv_policy::move)
        .def("add", [](Scene& scene, std::unique_ptr<MeshAsset> asset) {
            return scene.add(std::move(*asset));
        }, "asset"_a)
        .def("add", [](Scene& scene, std::unique_ptr<Texture> texture) -> Texture& {
            return scene.add(std::move(*texture));
        }, "texture"_a, nb::rv_policy::reference_internal)
        .def("remove", &Scene::removeObject, "object_id"_a)
        .def("reparent", &Scene::reparentObject, "object_id"_a, "new_parent_id"_a = 0)
        .def("get_object", &Scene::getObjectPtr, "object_id"_a)
        .def_prop_ro("objects", &Scene::getSceneObjects)
        .def_prop_ro("active_camera", &Scene::getActiveCameraPtr)
        .def_prop_rw("active_object_id", &Scene::getActiveObjectId, &Scene::setActiveObjectId)
        .def_prop_ro("environment", nb::overload_cast<>(&Scene::getEnvironment), nb::rv_policy::reference_internal)
        .def_prop_ro("render_settings", nb::overload_cast<>(&Scene::getRenderSettings), nb::rv_policy::reference_internal)
        .def("set_active_camera", &Scene::setActiveCamera, "camera"_a);
}
