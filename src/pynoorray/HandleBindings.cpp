#include "Bindings.h"

#include <string>

#include <nanobind/operators.h>
#include <nanobind/stl/string.h>

#include "Scene/Handle.h"
#include "Scene/SceneResources.h"

namespace nb = nanobind;

namespace
{
// Handles are opaque on the Python side: they compare, hash and print, but the
// index is exposed read-only because nothing outside the engine should be
// making one up.
template<class Tag>
void bindHandle(nb::module_& module, const char* name)
{
    using HandleType = nr::Handle<Tag>;
    nb::class_<HandleType>(module, name)
        .def(nb::init<>())
        .def_prop_ro("index", &HandleType::index)
        .def_prop_ro("generation", &HandleType::generation)
        .def_prop_ro("valid", &HandleType::isValid)
        .def("__bool__", &HandleType::isValid)
        .def(nb::self == nb::self)
        .def("__hash__", [](const HandleType& handle) {
            return std::hash<HandleType>{}(handle);
        })
        .def("__repr__", [name](const HandleType& handle) {
            return std::string(name) + "(" + std::to_string(handle.index())
                + ", " + std::to_string(handle.generation()) + ")";
        });
}

// References own their resource: holding one from Python keeps the asset (and
// its GPU memory) alive exactly as holding one from C++ does.
template<class Registry>
void bindRef(nb::module_& module, const char* name)
{
    using RefType = nr::ResourceRef<Registry>;
    nb::class_<RefType>(module, name)
        .def(nb::init<>())
        .def_prop_ro("handle", &RefType::handle)
        .def_prop_ro("index", &RefType::index)
        .def_prop_ro("valid", &RefType::isValid)
        .def("__bool__", &RefType::isValid)
        .def("release", &RefType::reset)
        .def(nb::self == nb::self);
}
}

void bindHandles(nb::module_& module)
{
    bindHandle<SceneObject>(module, "SceneObjectHandle");
    bindHandle<MeshAsset>(module, "MeshAssetHandle");
    bindHandle<GaussianAsset>(module, "GaussianAssetHandle");
    bindHandle<Material>(module, "MaterialHandle");
    bindHandle<Texture>(module, "TextureHandle");

    bindRef<MeshAssetRegistry>(module, "MeshAssetRef");
    bindRef<GaussianAssetRegistry>(module, "GaussianAssetRef");
    bindRef<MaterialRegistry>(module, "MaterialRef");
    bindRef<TextureRegistry>(module, "TextureRef");
}
