#include "Bindings.h"

#include <cstring>

#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include "IO/Bitmap.h"
#include "Vulkan/Image.h"

namespace nb = nanobind;
using namespace nb::literals;

void bindImage(nb::module_& module)
{
    nb::class_<Image>(module, "Image")
        .def_prop_ro("width", &Image::getWidth)
        .def_prop_ro("height", &Image::getHeight)
        .def("save", &Image::save, "path"_a)
        .def("numpy", [](const Image& image) {
            const Bitmap bitmap = image.toBitmap();
            const size_t valueCount = static_cast<size_t>(bitmap.width()) * bitmap.height() * 4;
            auto* pixels = new float[valueCount];
            std::memcpy(pixels, bitmap.rgba(), valueCount * sizeof(float));
            nb::capsule owner(pixels, [](void* pointer) noexcept {
                delete[] static_cast<float*>(pointer);
            });
            const size_t shape[3] = {bitmap.height(), bitmap.width(), 4};
            return nb::ndarray<nb::numpy, float, nb::shape<-1, -1, 4>>(
                pixels, 3, shape, owner);
        });
}
