#include "NoorRay.Python/Engine.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <format>
#include <mutex>
#include <thread>
#include <utility>

#include "NoorRay/Core/CacheDirectory.h"
#include "NoorRay/Core/Log.h"
#include "NoorRay/Format/AnimationRecorder.h"
#include "NoorRay/Format/SceneWriter.h"
#include "NoorRay/Render/Upscale/DlssRr.h"

namespace NoorRay::Python {

namespace {

//! The device and pipelines that the last closed engine left for the next one, and the thread that prepares the first.
struct IdleDevice {
    std::mutex mutex;
    NoorRHI::Device device;
    std::shared_ptr<Render::PipelineCache> pipelines;
    std::thread preparing;
};

IdleDevice& getIdleDevice() {
    static IdleDevice idle;
    return idle;
}

std::expected<void, Core::Error> makeDevice(NoorRHI::Device& device, std::shared_ptr<Render::PipelineCache>& pipelines) {
    std::expected<std::filesystem::path, Core::Error> cache = Core::getCacheDirectory();
    if (!cache)
        return std::unexpected(std::move(cache.error()));
    //! OpenDLSS-RR needs its Vulkan extensions when the device is made, so the device asks for them before the denoiser is known.
    static const Render::DlssRrExtensions extensions;
    if (auto opened = Core::check(device.open({.enableValidation = std::getenv("NOORRAY_VALIDATION") != nullptr,
                                               .applicationName = "NoorRay",
                                               .extensions = &extensions,
                                               .pipelineDirectory = *cache}));
        !opened)
        return opened;
    pipelines = std::make_shared<Render::PipelineCache>();
    return {};
}

//! Waits for the thread that prepares the idle device, outside the lock that the thread takes.
void finishPreparing(IdleDevice& idle) {
    std::unique_lock lock(idle.mutex);
    std::thread preparing = std::move(idle.preparing);
    lock.unlock();
    if (preparing.joinable())
        preparing.join();
}

}

std::expected<void, Core::Error> KeptDevice::open() {
    IdleDevice& idle = getIdleDevice();
    finishPreparing(idle);
    {
        std::lock_guard lock(idle.mutex);
        if (idle.device) {
            device = std::exchange(idle.device, {});
            pipelines = std::move(idle.pipelines);
            return {};
        }
    }
    return makeDevice(device, pipelines);
}

KeptDevice::~KeptDevice() {
    //! A device that cannot finish its work, as after a GPU fault, is not kept.
    if (!device || !device.waitIdle())
        return;
    IdleDevice& idle = getIdleDevice();
    std::lock_guard lock(idle.mutex);
    if (idle.device)
        return;
    idle.device = std::move(device);
    idle.pipelines = std::move(pipelines);
}

void prepareIdleDevice() {
    IdleDevice& idle = getIdleDevice();
    std::lock_guard lock(idle.mutex);
    if (idle.device || idle.preparing.joinable())
        return;
    idle.preparing = std::thread([&idle] {
        NoorRHI::Device device;
        std::shared_ptr<Render::PipelineCache> pipelines;
        //! The engine that opens next makes the device again and reports the error.
        if (auto made = makeDevice(device, pipelines); !made) {
            Core::Log::warning("the device could not be prepared: {}", made.error().message);
            return;
        }
        std::lock_guard lock(idle.mutex);
        if (idle.device)
            return;
        idle.device = std::move(device);
        idle.pipelines = std::move(pipelines);
    });
}

void releaseIdleDevice() {
    IdleDevice& idle = getIdleDevice();
    finishPreparing(idle);
    std::lock_guard lock(idle.mutex);
    idle.pipelines.reset();
    idle.device = {};
}

std::expected<void, Core::Error> Engine::open(const std::uint32_t width, const std::uint32_t height) {
    if (auto opened = kept.open(); !opened)
        return opened;
    return session.open(kept.device, width, height, kept.pipelines);
}

namespace {

//! The sync of the frame that Blender is at, as the capsule of its frame change handler.
class FrameExport {
public:
    FrameExport(const Object& depsgraph, const bool approximate) : depsgraph(depsgraph), approximate(approximate) {}
    FrameExport(const FrameExport&) = delete;
    FrameExport& operator=(const FrameExport&) = delete;

    //! Records each frame of the range.
    [[nodiscard]] std::expected<void, Core::Error> record(FrameRange range);
    Format::RecordedAnimation take(const float framesPerSecond) { return recorder.take(framesPerSecond); }
    std::vector<std::string> takeWarnings() { return std::move(warnings); }

private:
    static PyObject* onFrame(PyObject* capsule, PyObject* arguments);
    void sync();
    [[nodiscard]] std::expected<void, Core::Error> recordFrame(const Object& blenderScene, int target, bool first);
    [[nodiscard]] std::expected<void, Core::Error> setFrame(const Object& blenderScene);

    const Object& depsgraph;
    bool approximate;
    Reader reader;
    Scene::Scene scene;
    SceneSync blender;
    Format::AnimationRecorder recorder;
    std::vector<std::string> warnings;
    int frame = 0;
    bool synced = false;
    std::optional<Core::Error> error;
};

//! Blender clears the updates of a depsgraph after its frame change handlers, and it calls them once for the depsgraph of each view layer.
PyObject* FrameExport::onFrame(PyObject* capsule, PyObject* arguments) {
    auto& self = *static_cast<FrameExport*>(PyCapsule_GetPointer(capsule, nullptr));
    PyObject* scene = nullptr;
    PyObject* changed = nullptr;
    if (!PyArg_UnpackTuple(arguments, "frame_change_post", 2, 2, &scene, &changed))
        return nullptr;
    if (changed != Py_None && self.reader.pointer(Object::borrow(changed)) == self.reader.pointer(self.depsgraph))
        self.sync();
    Py_RETURN_NONE;
}

void FrameExport::sync() {
    if (error || synced)
        return;
    std::expected<void, Core::Error> read = blender.sync(scene, depsgraph, approximate);
    if (!read) {
        error = Core::Error{read.error().code, std::format("frame {}: {}", frame, read.error().message)};
        return;
    }
    for (std::string& warning : blender.takeWarnings())
        if (!std::ranges::contains(warnings, warning))
            warnings.push_back(std::move(warning));
    synced = true;
}

//! Blender switches to the camera of a timeline marker after the frame change handlers, and a first sync learns instance attributes while it reads instances.
std::expected<void, Core::Error> FrameExport::recordFrame(const Object& blenderScene, const int target, const bool first) {
    const auto camera = [&] {
        const Object current = reader.find(blenderScene, "camera");
        return current ? reader.pointer(current) : 0;
    };
    frame = target;
    const std::uintptr_t before = camera();
    std::expected<void, Core::Error> result = setFrame(blenderScene);
    if (result && (first || camera() != before))
        result = setFrame(blenderScene);
    if (!result)
        return result;
    recorder.record(scene, frame);
    scene.takeChange();
    return {};
}

std::expected<void, Core::Error> FrameExport::setFrame(const Object& blenderScene) {
    synced = false;
    reader.call(blenderScene, "frame_set", std::array{Object(PyLong_FromLong(frame))});
    if (std::expected<void, Core::Error> read = reader.takeError(); !read)
        return read;
    if (error)
        return std::unexpected(*error);
    if (!synced)
        return Core::fail(Core::ErrorCode::InvalidState, std::format("frame {} did not evaluate the depsgraph of the export", frame));
    return {};
}

std::expected<void, Core::Error> FrameExport::record(const FrameRange range) {
    static PyMethodDef method{"noorray_export_frame", onFrame, METH_VARARGS, nullptr};
    const Object capsule(PyCapsule_New(this, nullptr, nullptr));
    const Object handler(PyCFunction_New(&method, capsule.get()));
    const Object blenderScene = reader.get(depsgraph, "scene");
    const Object handlers = reader.get(reader.get(reader.get(reader.module("bpy"), "app"), "handlers"), "frame_change_post");
    const Object current(PyLong_FromLong(reader.integer(blenderScene, "frame_current")));
    const Object subframe(PyFloat_FromDouble(reader.number(blenderScene, "frame_subframe")));
    reader.call(handlers, "append", std::array{handler});
    std::expected<void, Core::Error> result = reader.takeError();
    if (!result)
        return result;
    for (int next = range.first; next <= range.last && result; ++next)
        result = recordFrame(blenderScene, next, next == range.first);
    reader.call(handlers, "remove", std::array{handler});
    reader.call(reader.get(blenderScene, "frame_set"), current, "subframe", subframe);
    std::expected<void, Core::Error> restored = reader.takeError();
    return result ? restored : result;
}

std::expected<std::vector<std::string>, Core::Error> exportAnimation(const Object& depsgraph, const std::filesystem::path& path, const bool approximate,
                                                                     const FrameRange frames) {
    Reader reader;
    const Object render = reader.get(reader.get(depsgraph, "scene"), "render");
    const float framesPerSecond = reader.number(render, "fps") / reader.number(render, "fps_base");
    if (std::expected<void, Core::Error> read = reader.takeError(); !read)
        return std::unexpected(std::move(read.error()));
    FrameExport frameExport(depsgraph, approximate);
    if (std::expected<void, Core::Error> recorded = frameExport.record(frames); !recorded)
        return std::unexpected(std::move(recorded.error()));
    const Format::RecordedAnimation recorded = frameExport.take(framesPerSecond);
    if (std::expected<void, Core::Error> written = Format::SceneWriter::write(recorded.scene, recorded.animation, path); !written)
        return std::unexpected(std::move(written.error()));
    return frameExport.takeWarnings();
}

}

std::expected<std::vector<std::string>, Core::Error> exportScene(const Object& depsgraph, const std::filesystem::path& path, const bool approximate,
                                                                 const std::optional<FrameRange> frames) {
    if (frames)
        return exportAnimation(depsgraph, path, approximate, *frames);
    Scene::Scene scene;
    SceneSync blender;
    if (std::expected<void, Core::Error> synced = blender.sync(scene, depsgraph, approximate); !synced)
        return std::unexpected(std::move(synced.error()));
    if (std::expected<void, Core::Error> written = Format::SceneWriter::write(scene, path); !written)
        return std::unexpected(std::move(written.error()));
    return blender.takeWarnings();
}

std::expected<void, Core::Error> Engine::sync(const Object& depsgraph, const bool approximate, const bool accumulates) {
    changed = true;
    session.getRenderer().setAccumulatesChanges(accumulates);
    std::expected<void, Core::Error> synced = blender.sync(session.getScene(), depsgraph, approximate);
    warnings = blender.takeWarnings();
    if (!synced)
        return synced;
    if (window)
        return crop(*window);
    if (!view)
        return {};
    //! A sync that cleared the scene adds the Blender camera again; an unchanged view changes nothing.
    return setCamera(*view);
}

std::expected<void, Core::Error> Engine::resize(const std::uint32_t width, const std::uint32_t height) {
    if (width == getWidth() && height == getHeight())
        return {};
    changed = true;
    return session.resize(width, height);
}

void Engine::configure(const EngineSettings& settings) {
    Render::RendererSettings renderer = Render::RendererSettings::pathTraced();
    renderer.maxSamples = settings.samples;
    renderer.denoiser = settings.denoiser;
    renderer.temporal = settings.temporal;
    //! The viewport renders each frame on its own, as in a game; a final render accumulates up to samples.
    renderer.accumulate = settings.finalRender;
    renderer.upscaler = settings.temporal && settings.upscale ? Render::Upscaler::Fsr : Render::Upscaler::Off;
    renderer.renderScale = settings.renderScale;
    renderer.sharpness = settings.sharpness;
    renderer.reuse = settings.reuse;
    renderer.radianceCache = settings.radianceCache;
    renderer.wavefront.lightTree = settings.lightTree;
    renderer.wavefront.visibilityCache = settings.visibilityCache;
    renderer.wavefront.gaussianShading = settings.gaussianShading;
    renderer.wavefront.seed = settings.seed;
    renderer.wavefront.compilesInBackground = !settings.finalRender;
    if (settings.denoiser == Render::Denoiser::OpenDlssRr) {
        //! DLSS-RR needs temporal frames (RendererSettings::validated); it replaces RELAX and FSR.
        renderer.temporal = true;
        renderer.upscaler = Render::Upscaler::Off;
    }
    //! DLSS-RR does not accumulate, so RELAX denoises a final render.
    if (settings.finalRender && renderer.denoiser == Render::Denoiser::OpenDlssRr)
        renderer.denoiser = Render::Denoiser::Nrd;
    accumulates = renderer.accumulate;
    samples = settings.samples;
    renderer.wavefront.maxDepth = settings.maxBounces + 1;
    renderer.wavefront.maxVolumeBounces = settings.volumeBounces;
    renderer.wavefront.maxTransparentBounces = settings.transparentBounces;
    renderer.wavefront.maxDiffuseBounces = settings.diffuseBounces;
    renderer.wavefront.maxGlossyBounces = settings.glossyBounces;
    renderer.wavefront.maxTransmissionBounces = settings.transmissionBounces;
    renderer.wavefront.filterGlossy = settings.filterGlossy;
    renderer.wavefront.exposure = settings.exposure;
    renderer.wavefront.directClamp = settings.clampDirect;
    renderer.wavefront.indirectClamp = settings.clampIndirect;
    renderer.wavefront.filterWidth = settings.filterWidth;
    renderer.display = session.getRenderer().getSettings().display;
    session.getRenderer().setSettings(renderer);
    changed = true;
}

std::expected<void, Core::Error> Engine::setCamera(const CameraView& view) {
    Scene::Scene& scene = session.getScene();
    if (!scene.findActiveCamera())
        return Core::fail(Core::ErrorCode::InvalidState, "the scene has no camera");
    Scene::Camera camera = blender.getCamera();
    camera.transform = view.transform;
    camera.previousTransform = view.transform;
    camera.pixelAspect = 1.0f;
    if (!view.throughCamera)
        camera.fStop = 0.0f;
    if (view.window)
        camera.window = *view.window;
    else
        setProjection(camera, view);
    scene.setCamera(scene.getActiveCamera(), camera);
    blender.keepCamera();
    this->view = view;
    changed = true;
    return {};
}

void Engine::setProjection(Scene::Camera& camera, const CameraView& view) const {
    //! The zoom of the view widens the sensor but keeps the lens, and with it the aperture.
    const float focal = camera.getFocalLength();
    camera.type = view.orthographic ? Scene::CameraType::Orthographic : Scene::CameraType::Perspective;
    camera.sensor.height = 2.0f * focal * std::tan(0.5f * view.verticalFov);
    camera.verticalFov = view.verticalFov;
    camera.orthographicHeight = view.orthographicHeight;
    const float aspect = float(getWidth()) / float(std::max(getHeight(), 1u));
    camera.sensor.width = camera.sensor.height * aspect;
    camera.sensor.offset = {0.5f * view.shift.x * camera.sensor.width, 0.5f * view.shift.y * camera.sensor.height};
    camera.near = view.near;
    camera.far = view.far;
}

std::expected<void, Core::Error> Engine::crop(const Core::float4 window) {
    Scene::Scene& scene = session.getScene();
    const Scene::Camera* active = scene.findActiveCamera();
    if (!active)
        return Core::fail(Core::ErrorCode::InvalidState, "the scene has no camera");
    Scene::Camera camera = *active;
    camera.window = window;
    scene.setCamera(scene.getActiveCamera(), camera);
    this->window = window;
    changed = true;
    return {};
}

std::expected<void, Core::Error> Engine::render() {
    Render::Renderer& renderer = session.getRenderer();
    const std::size_t slot = (sharedSlot + 1) % GlInterop::slotCount;
    const bool shares = gl.isOpen() && shared[slot].getWidth() == getWidth() && shared[slot].getHeight() == getHeight();
    if (shares)
        if (auto waited = gl.waitForCopy(slot); !waited)
            return waited;
    renderer.setSharedOutput(shares ? &shared[slot] : nullptr);
    NoorRHI::CommandList list(kept.device);
    if (changed)
        frames = 0;
    if (auto recorded = session.record(list); !recorded)
        return recorded;
    if (changed)
        session.getScene().releaseTextureData();
    changed = false;
    ++frames;
    //! The CPU records the next frame while the GPU renders this one; it waits only when more than slotCount frames are in flight.
    auto token = Core::check(kept.device.submit(std::move(list)));
    if (!token)
        return std::unexpected(std::move(token.error()));
    while (!inFlight.empty() && kept.device.isFinished(inFlight.front()))
        inFlight.pop_front();
    inFlight.push_back(*token);
    for (; inFlight.size() > GlInterop::slotCount; inFlight.pop_front())
        if (auto waited = Core::check(kept.device.wait(inFlight.front())); !waited)
            return waited;
    if (renderer.wroteSharedOutput()) {
        sharedSlot = slot;
        sharedPending = true;
    }
    return {};
}

std::expected<void, Core::Error> Engine::denoise() {
    NoorRHI::CommandList list(kept.device);
    if (auto recorded = session.getRenderer().recordFilmDenoise(list); !recorded)
        return recorded;
    return submit(std::move(list));
}

std::expected<void, Core::Error> Engine::read(const std::span<float> pixels) {
    Render::Renderer& renderer = session.getRenderer();
    const std::size_t size = std::size_t{getWidth()} * getHeight();
    if (pixels.size() != size * 4)
        return Core::fail(Core::ErrorCode::InvalidArgument, "the pixel buffer does not have four floats for each pixel of the image");
    NoorRHI::CommandList list(kept.device);
    //! FSR and DLSS-RR resolve to the display size into their own image; the output of the targets then has the render size.
    const NoorRHI::Buffer<NoorRHI::float4>* source = &renderer.getTargets().getOutput();
    if (renderer.getFrameSettings().upscales()) {
        if (!upscaled || upscaled.getSize() != size)
            if (auto opened = Core::check(upscaled.open(kept.device, size)); !opened)
                return opened;
        if (auto copied = renderer.recordUpscaledCopy(list, upscaled); !copied)
            return copied;
        source = &upscaled;
    }
    if (!film.getBackend() || film.getSize() != size)
        if (auto opened = Core::check(film.open(kept.device, size)); !opened)
            return opened;
    list.download(*source, film);
    if (auto submitted = submit(std::move(list)); !submitted)
        return submitted;
    auto values = Core::check(film.read());
    if (!values)
        return std::unexpected(values.error());
    static_assert(sizeof(NoorRHI::float4) == 4 * sizeof(float));
    std::memcpy(pixels.data(), values->data(), pixels.size_bytes());
    return {};
}

std::expected<void, Core::Error> Engine::readPass(const FilmPass pass, const std::span<float> values) {
    const Render::Targets& targets = session.getRenderer().getTargets();
    const std::size_t width = targets.getWidth();
    const std::size_t count = targets.getPixelCount();
    const std::size_t channels = pass == FilmPass::Depth ? 1 : 3;
    if (values.size() != count * channels)
        return Core::fail(Core::ErrorCode::InvalidArgument, "the pass buffer does not have the size of the film");
    const NoorRHI::Buffer<NoorRHI::float4>& source = pass == FilmPass::DiffuseColor ? targets.getAlbedo() : targets.getNormalDepth();
    if (!film.getBackend() || film.getSize() != count)
        if (auto opened = Core::check(film.open(kept.device, count)); !opened)
            return opened;
    NoorRHI::CommandList list(kept.device);
    list.download(source, film);
    if (auto submitted = submit(std::move(list)); !submitted)
        return submitted;
    auto read = Core::check(film.read());
    if (!read)
        return std::unexpected(read.error());
    const Shaders::CameraRecord& camera = session.getRenderer().getCameraRecord();
    const bool perspective = camera.meta.x == Shaders::cameraTypePerspective;
    for (std::size_t index = 0; index < count; ++index) {
        const NoorRHI::float4 value = (*read)[index];
        const bool hit = value.w > 0.0f;
        if (pass == FilmPass::Depth) {
            const float x = (float(index % width) + 0.5f) / float(width) - 0.5f;
            const float y = 0.5f - (float(index / width) + 0.5f) / float(targets.getHeight());
            const float focal = camera.projection.x;
            const float cosine = focal / std::hypot(x * camera.sensor.x + camera.sensor.z, y * camera.sensor.y + camera.sensor.w, focal);
            //! The rays start at the near plane, and Blender's depth of a pixel without a surface is 1e10.
            values[index] = hit ? value.w * (perspective ? cosine : 1.0f) + camera.clip.x : 1e10f;
        } else if (pass == FilmPass::Normal) {
            values[index * 3] = hit ? value.x : 0.0f;
            values[index * 3 + 1] = hit ? -value.z : 0.0f;
            values[index * 3 + 2] = hit ? value.y : 0.0f;
        } else {
            values[index * 3] = value.x;
            values[index * 3 + 1] = value.y;
            values[index * 3 + 2] = value.z;
        }
    }
    return {};
}

std::expected<void, Core::Error> Engine::openGl() {
    const NoorRHI::ImageDesc desc{
        .width = getWidth(),
        .height = getHeight(),
        .format = NoorRHI::ImageFormat::Rgba16Float,
        .usage = NoorRHI::ImageUsage::Storage | NoorRHI::ImageUsage::ExternalMemory,
        .name = "NoorRay OpenGL film",
    };
    for (NoorRHI::Image& image : shared)
        if (auto opened = Core::check(image.open(kept.device, desc)); !opened)
            return opened;
    sharedPending = false;
    //! The new images hold no frame, so the next render traces one even when the film was complete.
    session.getRenderer().restartAccumulation();
    changed = true;
    return gl.open(kept.device, shared);
}

std::expected<void, Core::Error> Engine::presentGl() {
    if (!gl.isOpen())
        return Core::fail(Core::ErrorCode::InvalidState, "the OpenGL film is not open");
    if (!sharedPending)
        return {};
    sharedPending = false;
    return gl.present(sharedSlot);
}

std::expected<void, Core::Error> Engine::submit(NoorRHI::CommandList&& list) {
    auto token = Core::check(kept.device.submit(std::move(list)));
    if (!token)
        return std::unexpected(token.error());
    return Core::check(kept.device.wait(*token));
}

}
