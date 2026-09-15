#include "Bindings.h"

#include "NoorRaySession.h"
#include <nanobind/ndarray.h>

#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

void bindNoorRaySession(nb::module_& module)
{
    nb::class_<noorray::NoorRaySession>(module, "NoorRaySession")
        .def(nb::init<>())
        .def_prop_ro("scene", [](noorray::NoorRaySession& session) -> Scene& {
            return session.scene();
        }, nb::rv_policy::reference_internal)
        .def("initialize_renderer", &noorray::NoorRaySession::initializeHeadlessRenderer,
            "width"_a, "height"_a, "export_color_memory"_a = false)
        .def("rebuild_renderer", &noorray::NoorRaySession::rebuildNativeScene)
        .def("render", [](noorray::NoorRaySession& session,
             const std::uint32_t frame_index, const std::uint32_t sample_index) {
            if (!session.hasRenderer())
                throw std::runtime_error("initialize_renderer must be called before render");
            session.render(frame_index, sample_index);
            session.synchronize();
        }, "frame_index"_a = 0u, "sample_index"_a = 0u)
        .def("beauty_bgra", [](noorray::NoorRaySession& session) {
            if (!session.hasRenderer())
                throw std::runtime_error("initialize_renderer must be called before beauty_bgra");
            std::vector<std::byte> bytes = session.readColor();
            auto* data = new std::uint8_t[bytes.size()];
            std::memcpy(data, bytes.data(), bytes.size());
            nb::capsule owner(data, [](void* pointer) noexcept { delete[] static_cast<std::uint8_t*>(pointer); });
            const size_t shape[3] = {session.outputHeight(), session.outputWidth(), 4};
            return nb::ndarray<nb::numpy, std::uint8_t, nb::shape<-1, -1, 4>>(data, 3, shape, owner);
        })
        .def("beauty_rgba", [](noorray::NoorRaySession& session) {
            if (!session.hasRenderer())
                throw std::runtime_error("initialize_renderer must be called before beauty_rgba");
            const std::vector<noorrhi::float4> pixels = session.readBeauty();
            auto* data = new float[pixels.size() * 4];
            std::memcpy(data, pixels.data(), pixels.size() * sizeof(noorrhi::float4));
            nb::capsule owner(data, [](void* pointer) noexcept { delete[] static_cast<float*>(pointer); });
            const size_t shape[3] = {session.outputHeight(), session.outputWidth(), 4};
            return nb::ndarray<nb::numpy, float, nb::shape<-1, -1, 4>>(data, 3, shape, owner);
        })
        .def_ro("headless", &noorray::NoorRaySession::isHeadless);
}
