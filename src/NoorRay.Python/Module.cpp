#include <cstring>
#include <filesystem>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include <Python.h>

#include "NoorRay.Python/Engine.h"
#include "NoorRay/Render/Denoise/DenoiserPlugin.h"
#include "NoorRay/Render/Upscale/DlssRr.h"

namespace {

using NoorRay::Python::CameraView;
using NoorRay::Python::Engine;
using NoorRay::Python::EngineSettings;
using NoorRay::Python::FilmPass;
namespace Core = NoorRay::Core;

//! NRD (RELAX), or the OpenDLSS-RR plugin.
std::optional<NoorRay::Render::Denoiser> parseDenoiser(const std::string_view name) {
    using NoorRay::Render::Denoiser;
    if (name == "off")
        return Denoiser::Off;
    if (name == "nrd")
        return Denoiser::Nrd;
    if (name == "open_dlss_rr")
        return Denoiser::OpenDlssRr;
    return std::nullopt;
}

std::optional<NoorRay::Render::Reuse> parseReuse(const std::string_view name) {
    using NoorRay::Render::Reuse;
    if (name == "off")
        return Reuse::Off;
    if (name == "di")
        return Reuse::Di;
    if (name == "pt")
        return Reuse::Pt;
    return std::nullopt;
}

std::optional<NoorRay::Render::GaussianShading> parseGaussianShading(const std::string_view name) {
    using NoorRay::Render::GaussianShading;
    if (name == "off")
        return GaussianShading::Off;
    if (name == "emissive")
        return GaussianShading::Emissive;
    if (name == "scattering")
        return GaussianShading::Scattering;
    return std::nullopt;
}

bool isBuilt(const NoorRay::Render::Denoiser denoiser) {
    return denoiser != NoorRay::Render::Denoiser::OpenDlssRr || NoorRay::Render::DlssRr::isBuilt();
}

struct EngineObject {
    PyObject_HEAD std::unique_ptr<Engine> engine;
};

Engine& engineOf(PyObject* self) {
    return *reinterpret_cast<EngineObject*>(self)->engine;
}

PyObject* raise(const NoorRay::Core::Error& error) {
    PyErr_SetString(PyExc_RuntimeError, error.message.c_str());
    return nullptr;
}

PyObject* result(const std::expected<void, NoorRay::Core::Error>& outcome) {
    if (!outcome)
        return raise(outcome.error());
    Py_RETURN_NONE;
}

template <class Function>
auto withoutGil(Function&& function) {
    PyThreadState* state = PyEval_SaveThread();
    auto value = function();
    PyEval_RestoreThread(state);
    return value;
}

class BufferView {
public:
    ~BufferView() {
        if (opened)
            PyBuffer_Release(&view);
    }

    bool open(PyObject* object, const int flags) {
        opened = PyObject_GetBuffer(object, &view, flags) == 0;
        return opened;
    }

    Py_buffer view{};

private:
    bool opened = false;
};

PyObject* newEngine(PyTypeObject* type, PyObject*, PyObject*) {
    auto* allocate = reinterpret_cast<allocfunc>(PyType_GetSlot(type, Py_tp_alloc));
    PyObject* self = allocate(type, 0);
    if (!self)
        return nullptr;
    new (&reinterpret_cast<EngineObject*>(self)->engine) std::unique_ptr<Engine>(std::make_unique<Engine>());
    return self;
}

void deleteEngine(PyObject* self) {
    PyTypeObject* type = Py_TYPE(self);
    std::destroy_at(&reinterpret_cast<EngineObject*>(self)->engine);
    auto* release = reinterpret_cast<freefunc>(PyType_GetSlot(type, Py_tp_free));
    release(self);
    Py_DECREF(type);
}

int initEngine(PyObject* self, PyObject* arguments, PyObject* keywords) {
    static const char* names[] = {"width", "height", nullptr};
    unsigned int width = 0;
    unsigned int height = 0;
    if (!PyArg_ParseTupleAndKeywords(arguments, keywords, "II", const_cast<char**>(names), &width, &height))
        return -1;
    const std::expected<void, NoorRay::Core::Error> opened = withoutGil([&] { return engineOf(self).open(width, height); });
    if (opened)
        return 0;
    raise(opened.error());
    return -1;
}

PyObject* sync(PyObject* self, PyObject* arguments, PyObject* keywords) {
    static const char* names[] = {"depsgraph", "approximate", "accumulates", nullptr};
    PyObject* depsgraph = nullptr;
    int approximate = 0;
    int accumulates = 0;
    if (!PyArg_ParseTupleAndKeywords(arguments, keywords, "O|$pp", const_cast<char**>(names), &depsgraph, &approximate, &accumulates))
        return nullptr;
    return result(engineOf(self).sync(NoorRay::Python::Object::borrow(depsgraph), approximate != 0, accumulates != 0));
}

PyObject* resize(PyObject* self, PyObject* arguments) {
    unsigned int width = 0;
    unsigned int height = 0;
    if (!PyArg_ParseTuple(arguments, "II", &width, &height))
        return nullptr;
    return result(engineOf(self).resize(width, height));
}

PyObject* configure(PyObject* self, PyObject* arguments, PyObject* keywords) {
    static const char* names[] = {"samples",
                                  "denoiser",
                                  "temporal",
                                  "max_bounces",
                                  "volume_bounces",
                                  "transparent_bounces",
                                  "diffuse_bounces",
                                  "glossy_bounces",
                                  "transmission_bounces",
                                  "filter_glossy",
                                  "clamp_direct",
                                  "clamp_indirect",
                                  "filter_width",
                                  "final",
                                  "light_tree",
                                  "visibility_cache",
                                  "seed",
                                  "reuse",
                                  "radiance_cache",
                                  "upscale",
                                  "render_scale",
                                  "sharpness",
                                  "exposure",
                                  "gaussian_shading",
                                  nullptr};
    EngineSettings settings;
    const char* denoiser = NoorRay::Render::DlssRr::isBuilt() ? "open_dlss_rr" : "nrd";
    int temporal = settings.temporal;
    int finalRender = settings.finalRender;
    int lightTree = settings.lightTree;
    int visibilityCache = settings.visibilityCache;
    const char* reuse = "off";
    const char* gaussianShading = "emissive";
    int radianceCache = settings.radianceCache;
    int upscale = settings.upscale;
    if (!PyArg_ParseTupleAndKeywords(arguments, keywords, "|$IspIIIIIIffffpppIsppfffs", const_cast<char**>(names), &settings.samples, &denoiser, &temporal,
                                     &settings.maxBounces, &settings.volumeBounces, &settings.transparentBounces, &settings.diffuseBounces,
                                     &settings.glossyBounces, &settings.transmissionBounces, &settings.filterGlossy, &settings.clampDirect,
                                     &settings.clampIndirect, &settings.filterWidth, &finalRender, &lightTree, &visibilityCache, &settings.seed, &reuse,
                                     &radianceCache, &upscale, &settings.renderScale, &settings.sharpness, &settings.exposure, &gaussianShading))
        return nullptr;
    const std::optional<NoorRay::Render::Denoiser> parsed = parseDenoiser(denoiser);
    if (!parsed) {
        PyErr_Format(PyExc_ValueError, "the denoiser must be off, nrd or open_dlss_rr, not %s", denoiser);
        return nullptr;
    }
    if (!isBuilt(*parsed)) {
        PyErr_Format(PyExc_RuntimeError, "this build of NoorRay has no %s denoiser", denoiser);
        return nullptr;
    }
    settings.denoiser = *parsed;
    settings.lightTree = lightTree != 0;
    settings.visibilityCache = visibilityCache != 0;
    const std::optional<NoorRay::Render::Reuse> parsedReuse = parseReuse(reuse);
    if (!parsedReuse) {
        PyErr_Format(PyExc_ValueError, "the reuse must be off, di or pt, not %s", reuse);
        return nullptr;
    }
    settings.reuse = *parsedReuse;
    const std::optional<NoorRay::Render::GaussianShading> parsedGaussianShading = parseGaussianShading(gaussianShading);
    if (!parsedGaussianShading) {
        PyErr_Format(PyExc_ValueError, "the gaussian_shading must be off, emissive or scattering, not %s", gaussianShading);
        return nullptr;
    }
    settings.gaussianShading = *parsedGaussianShading;
    settings.radianceCache = radianceCache != 0;
    settings.upscale = upscale != 0;
    settings.temporal = temporal != 0;
    settings.finalRender = finalRender != 0;
    engineOf(self).configure(settings);
    Py_RETURN_NONE;
}

constexpr const char* transformFormat = "(ffffffffffff)";

bool parseTransform(PyObject* rows, NoorRay::Core::Transform& transform) {
    float (&r)[3][4] = transform.rows;
    PyObject* arguments = PyTuple_Pack(1, rows);
    if (!arguments)
        return false;
    const bool parsed = PyArg_ParseTuple(arguments, transformFormat, &r[0][0], &r[0][1], &r[0][2], &r[0][3], &r[1][0], &r[1][1], &r[1][2], &r[1][3], &r[2][0],
                                         &r[2][1], &r[2][2], &r[2][3]);
    Py_DECREF(arguments);
    return parsed;
}

PyObject* setCamera(PyObject* self, PyObject* arguments, PyObject* keywords) {
    static const char* names[] = {"transform",      "orthographic", "vertical_fov", "orthographic_height", "shift_x", "shift_y", "near", "far",
                                  "through_camera", "window",       nullptr};
    CameraView view;
    PyObject* rows = nullptr;
    int orthographic = 0;
    int throughCamera = 0;
    Core::float4 window{0.0f, 0.0f, 0.0f, 0.0f};
    if (!PyArg_ParseTupleAndKeywords(arguments, keywords, "O|$pffffffp(ffff)", const_cast<char**>(names), &rows, &orthographic, &view.verticalFov,
                                     &view.orthographicHeight, &view.shift.x, &view.shift.y, &view.near, &view.far, &throughCamera, &window.x, &window.y,
                                     &window.z, &window.w))
        return nullptr;
    view.throughCamera = throughCamera != 0;
    if (window.z != window.x)
        view.window = window;
    if (!parseTransform(rows, view.transform))
        return nullptr;
    view.orthographic = orthographic != 0;
    return result(engineOf(self).setCamera(view));
}

PyObject* crop(PyObject* self, PyObject* arguments) {
    Core::float4 window{0.0f, 0.0f, 1.0f, 1.0f};
    if (!PyArg_ParseTuple(arguments, "ffff", &window.x, &window.y, &window.z, &window.w))
        return nullptr;
    return result(engineOf(self).crop(window));
}

PyObject* render(PyObject* self, PyObject*) {
    return result(withoutGil([&] { return engineOf(self).render(); }));
}

PyObject* denoise(PyObject* self, PyObject*) {
    return result(withoutGil([&] { return engineOf(self).denoise(); }));
}

std::optional<std::span<float>> floats(BufferView& buffer, PyObject* pixels) {
    if (!buffer.open(pixels, PyBUF_CONTIG | PyBUF_FORMAT))
        return std::nullopt;
    if (buffer.view.itemsize != sizeof(float) || (buffer.view.format && std::strcmp(buffer.view.format, "f") != 0)) {
        PyErr_SetString(PyExc_TypeError, "the pixel buffer must hold 32-bit floats");
        return std::nullopt;
    }
    return std::span<float>(static_cast<float*>(buffer.view.buf), static_cast<std::size_t>(buffer.view.len) / sizeof(float));
}

PyObject* read(PyObject* self, PyObject* pixels) {
    BufferView buffer;
    const std::optional<std::span<float>> values = floats(buffer, pixels);
    if (!values)
        return nullptr;
    return result(withoutGil([&] { return engineOf(self).read(*values); }));
}

std::optional<FilmPass> parsePass(const std::string_view name) {
    if (name == "Depth")
        return FilmPass::Depth;
    if (name == "Normal")
        return FilmPass::Normal;
    if (name == "Diffuse Color")
        return FilmPass::DiffuseColor;
    return std::nullopt;
}

PyObject* readPass(PyObject* self, PyObject* arguments) {
    const char* name = nullptr;
    PyObject* pixels = nullptr;
    if (!PyArg_ParseTuple(arguments, "sO", &name, &pixels))
        return nullptr;
    const std::optional<FilmPass> pass = parsePass(name);
    if (!pass) {
        PyErr_Format(PyExc_ValueError, "there is no pass %s", name);
        return nullptr;
    }
    BufferView buffer;
    const std::optional<std::span<float>> values = floats(buffer, pixels);
    if (!values)
        return nullptr;
    return result(withoutGil([&] { return engineOf(self).readPass(*pass, *values); }));
}

PyObject* openGl(PyObject* self, PyObject*) {
    return result(engineOf(self).openGl());
}

PyObject* presentGl(PyObject* self, PyObject*) {
    return result(engineOf(self).presentGl());
}

PyObject* getComplete(PyObject* self, void*) {
    return PyBool_FromLong(engineOf(self).isComplete());
}

PyObject* getSamples(PyObject* self, void*) {
    return PyLong_FromUnsignedLong(engineOf(self).getSampleCount());
}

PyObject* getCompiling(PyObject* self, void*) {
    const std::string notice = engineOf(self).getCompileNotice();
    return PyUnicode_FromStringAndSize(notice.data(), static_cast<Py_ssize_t>(notice.size()));
}

PyObject* getWidth(PyObject* self, void*) {
    return PyLong_FromUnsignedLong(engineOf(self).getWidth());
}

PyObject* getHeight(PyObject* self, void*) {
    return PyLong_FromUnsignedLong(engineOf(self).getHeight());
}

PyObject* stringList(const std::span<const std::string> warnings) {
    PyObject* list = PyList_New(static_cast<Py_ssize_t>(warnings.size()));
    if (!list)
        return nullptr;
    for (std::size_t index = 0; index < warnings.size(); ++index) {
        PyObject* text = PyUnicode_FromStringAndSize(warnings[index].data(), static_cast<Py_ssize_t>(warnings[index].size()));
        if (!text) {
            Py_DECREF(list);
            return nullptr;
        }
        PyList_SetItem(list, static_cast<Py_ssize_t>(index), text);
    }
    return list;
}

PyObject* getWarnings(PyObject* self, void*) {
    return stringList(engineOf(self).getWarnings());
}

PyObject* getDenoisers(PyObject*, PyObject*) {
    PyObject* list = PyList_New(0);
    if (!list)
        return nullptr;
    for (const char* name : {"off", "nrd", "open_dlss_rr"})
        if (isBuilt(*parseDenoiser(name))) {
            PyObject* text = PyUnicode_FromString(name);
            if (!text || PyList_Append(list, text) != 0) {
                Py_XDECREF(text);
                Py_DECREF(list);
                return nullptr;
            }
            Py_DECREF(text);
        }
    return list;
}

PyObject* exportScene(PyObject*, PyObject* arguments, PyObject* keywords) {
    static const char* names[] = {"depsgraph", "path", "approximate", "frames", nullptr};
    PyObject* depsgraph = nullptr;
    const char* path = nullptr;
    int approximate = 0;
    PyObject* range = Py_None;
    if (!PyArg_ParseTupleAndKeywords(arguments, keywords, "Os|$pO", const_cast<char**>(names), &depsgraph, &path, &approximate, &range))
        return nullptr;
    std::optional<NoorRay::Python::FrameRange> frames;
    if (range != Py_None) {
        frames.emplace();
        if (!PyTuple_Check(range)) {
            PyErr_SetString(PyExc_TypeError, "frames must be None or a tuple of the first and the last frame");
            return nullptr;
        }
        if (!PyArg_ParseTuple(range, "ii", &frames->first, &frames->last))
            return nullptr;
        if (frames->last < frames->first) {
            PyErr_SetString(PyExc_ValueError, "the frame range ends before it starts");
            return nullptr;
        }
    }
    const std::expected<std::vector<std::string>, NoorRay::Core::Error> warnings = NoorRay::Python::exportScene(
        NoorRay::Python::Object::borrow(depsgraph), std::filesystem::path(reinterpret_cast<const char8_t*>(path)), approximate != 0, frames);
    if (!warnings)
        return raise(warnings.error());
    return stringList(*warnings);
}

PyObject* prepare(PyObject*, PyObject*) {
    NoorRay::Python::prepareIdleDevice();
    Py_RETURN_NONE;
}

PyObject* setHelpers(PyObject*, PyObject* module) {
    NoorRay::Python::Reader::setHelpers(NoorRay::Python::Object::borrow(module));
    Py_RETURN_NONE;
}

PyMethodDef moduleMethods[] = {
    {"set_helpers", setHelpers, METH_O, "Sets the Python module with the helper functions that a sync calls, or None."},
    {"export_scene", reinterpret_cast<PyCFunction>(reinterpret_cast<void (*)()>(exportScene)), METH_VARARGS | METH_KEYWORDS,
     "Writes a depsgraph as a .nrs file: export_scene(depsgraph, path, *, approximate=False, frames=None). frames is None for the current frame, or the first "
     "and the last frame of an animation. Returns the warnings."},
    {"denoisers", getDenoisers, METH_NOARGS,
     "The denoisers of this build: off, nrd and open_dlss_rr; a build without the OpenDLSS-RR plugin, such as the web build, has no open_dlss_rr."},
    {"prepare", prepare, METH_NOARGS, "Makes the Vulkan device on a thread, so that the first Engine opens without the wait."},
    {nullptr, nullptr, 0, nullptr},
};

PyMethodDef engineMethods[] = {
    {"sync", reinterpret_cast<PyCFunction>(reinterpret_cast<void (*)()>(sync)), METH_VARARGS | METH_KEYWORDS,
     "Brings the scene to a Blender depsgraph: sync(depsgraph, *, approximate=False, accumulates=False). Unsupported nodes fail unless approximate is set; "
     "accumulates keeps the samples of the film."},
    {"resize", resize, METH_VARARGS, "Resizes the film to width x height."},
    {"configure", reinterpret_cast<PyCFunction>(reinterpret_cast<void (*)()>(configure)), METH_VARARGS | METH_KEYWORDS,
     "Sets the path tracer settings: samples (the frames of a final render; the samples of a still viewport, 0 has no limit), denoiser (see denoisers()), "
     "temporal, max_bounces, "
     "volume_bounces, transparent_bounces, clamp_direct and clamp_indirect (0 for none), filter_width (pixels of the Blackman-Harris pixel filter), "
     "final (wait for the kernels), light_tree (false picks lights by power), "
     "visibility_cache, seed, reuse (off, di for ReSTIR DI or pt for ReSTIR PT), radiance_cache, upscale (FSR), render_scale (0.25 to 1, of FSR and DLSS-RR), "
     "sharpness, exposure and "
     "gaussian_shading (off, emissive or scattering)."},
    {"set_camera", reinterpret_cast<PyCFunction>(reinterpret_cast<void (*)()>(setCamera)), METH_VARARGS | METH_KEYWORDS,
     "Sets the active camera; every call restarts the accumulation."},
    {"crop", crop, METH_VARARGS, "Sets the part of the image that the film covers: crop(left, top, right, bottom), from 0 to 1 with the top at 0."},
    {"read_pass", readPass, METH_VARARGS,
     "Writes a geometry pass of the first samples, top row first: read_pass(name, buffer), with Depth, Normal or Diffuse Color."},
    {"render", render, METH_NOARGS, "Renders one frame and waits for it."},
    {"denoise", denoise, METH_NOARGS, "Denoises the film of a final render that stopped before its samples; a complete film is denoised already."},
    {"read", read, METH_O, "Writes the linear RGBA film, top row first, into a writable buffer of width * height * 4 floats."},
    {"open_gl", openGl, METH_NOARGS,
     "Shares the film with the current OpenGL context. Draw an RGBA16F GPUTexture of the film size just before: it becomes the target of present_gl."},
    {"present_gl", presentGl, METH_NOARGS, "Copies the last rendered frame into the target texture of open_gl on the GPU, top row first, without the CPU."},
    {nullptr, nullptr, 0, nullptr},
};

PyGetSetDef engineProperties[] = {
    {"complete", getComplete, nullptr, "True when the film has the configured number of samples.", nullptr},
    {"samples", getSamples, nullptr, "The number of samples in the film.", nullptr},
    {"compiling", getCompiling, nullptr, "What still compiles in the background and what the frames show until it is ready; empty when nothing does.", nullptr},
    {"width", getWidth, nullptr, "The width of the film.", nullptr},
    {"height", getHeight, nullptr, "The height of the film.", nullptr},
    {"warnings", getWarnings, nullptr, "The warnings of the last sync.", nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr},
};

PyType_Slot engineSlots[] = {
    {Py_tp_new, reinterpret_cast<void*>(newEngine)},
    {Py_tp_init, reinterpret_cast<void*>(initEngine)},
    {Py_tp_dealloc, reinterpret_cast<void*>(deleteEngine)},
    {Py_tp_methods, engineMethods},
    {Py_tp_getset, engineProperties},
    {Py_tp_doc,
     const_cast<char*>("Engine(width, height): a Vulkan device, a NoorRay scene and its path tracer; a closed engine leaves its device to the next.")},
    {0, nullptr},
};

PyType_Spec engineSpec = {"noorray.Engine", sizeof(EngineObject), 0, Py_TPFLAGS_DEFAULT, engineSlots};

int executeModule(PyObject* module) {
    PyObject* type = PyType_FromSpec(&engineSpec);
    if (!type)
        return -1;
    const int added = PyModule_AddObjectRef(module, "Engine", type);
    Py_DECREF(type);
    return added;
}

PyModuleDef_Slot moduleSlots[] = {
    {Py_mod_exec, reinterpret_cast<void*>(executeModule)},
    {0, nullptr},
};

//! The idle device of the closed engines ends with the module, before Python ends.
void freeModule(void*) {
    NoorRay::Python::Reader::setHelpers({});
    NoorRay::Python::releaseIdleDevice();
}

PyModuleDef moduleDefinition = {PyModuleDef_HEAD_INIT, "noorray", "The NoorRay path tracer.", 0, moduleMethods, moduleSlots, nullptr, nullptr, freeModule};

}

PyMODINIT_FUNC PyInit_noorray() {
    NoorRay::Render::prepareDenoiserPlugins();
    return PyModuleDef_Init(&moduleDefinition);
}
