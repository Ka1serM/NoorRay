#include "Bindings.h"

#include <stdexcept>
#include <utility>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>

#include "Camera/Sensor.h"

namespace nb = nanobind;
using namespace nb::literals;

void bindSensor(nb::module_& module)
{
    nb::enum_<SensorType>(module, "SensorType")
        .value("RECTANGULAR", SensorType::Rectangular)
        .value("SCATTER_PSF", SensorType::ScatterPsf)
        .value("GATHER_PSF", SensorType::GatherPsf);

    nb::class_<Sensor>(module, "Sensor")
        .def_prop_rw("resolution",
            [](const Sensor& sensor) { return nb::make_tuple(sensor.resolutionX(), sensor.resolutionY()); },
            [](Sensor& sensor, const std::pair<uint32_t, uint32_t>& resolution) {
                sensor.setResolution(resolution.first, resolution.second);
            })
        .def_prop_rw("dimensions_mm",
            [](const Sensor& sensor) { return nb::make_tuple(sensor.width(), sensor.height()); },
            [](Sensor& sensor, const std::pair<float, float>& dimensions) {
                sensor.setDimensionsMm(dimensions.first, dimensions.second);
            })
        .def_prop_rw("image_sensor_path",
            [](const Sensor& sensor) { return std::string(sensor.getImageSensorPath()); },
            [](Sensor& sensor, const std::string& path) { sensor.setImageSensorPath(path); })
        .def_prop_ro("type", &Sensor::getType)
        .def_prop_rw("psf_grid_path", &Sensor::getPsfGridPath, &Sensor::setPsfGridPath)
        .def("load", [](Sensor& sensor, const std::string& path) {
            sensor.setImageSensorPath(path);
            if (!sensor.loadImageSensorDimensions())
                throw std::runtime_error("Failed to load image sensor: " + path);
        }, "path"_a)
        .def("load_psf_grid", &Sensor::loadPsfGrid, "path"_a);
}
