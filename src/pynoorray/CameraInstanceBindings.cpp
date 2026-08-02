#include "Bindings.h"

#include <utility>

#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include "Rendering/Camera/CameraInstance.h"
#include "Rendering/Camera/PerspectiveCamera.h"
#include "Rendering/Camera/ThinLensCamera.h"
#include "Rendering/Camera/OrthographicCamera.h"
#include "Rendering/Camera/FisheyeCamera.h"
#include "Rendering/Camera/RealisticCamera.h"
#include "Rendering/Camera/HybridPsfCamera.h"

namespace nb = nanobind;
using namespace nb::literals;

void bindCameraInstance(nb::module_& module)
{
    nb::class_<CameraInstance, SceneObject>(module, "CameraInstance")
        .def(nb::new_([](std::unique_ptr<Camera> camera, const std::string& name,
                        Transform transform) {
            nb::object pythonCamera = nb::find(camera.get());
            auto* instance = new CameraInstance(std::move(camera), name, transform);
            if (pythonCamera.is_valid())
                nb::inst_set_state(pythonCamera, true, false);
            return instance;
        }),
            "camera"_a, "name"_a = "Camera", "transform"_a = Transform{})
        .def_prop_ro("camera", nb::overload_cast<>(&CameraInstance::getCamera), nb::rv_policy::reference_internal)
        .def_prop_ro("projection_type", &CameraInstance::getProjectionType)
        .def("switch_to", &CameraInstance::switchTo, "projection_type"_a);
}
