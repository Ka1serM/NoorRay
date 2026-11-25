#include "NoorRay.Python/GlInterop.h"

#include <algorithm>
#include <array>
#include <format>
#include <optional>
#include <ranges>
#include <set>
#include <string_view>
#include <vector>

#include "NoorRHI/Device/Interop.h"

#if defined(_WIN32)
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#endif

namespace NoorRay::Python {

namespace {

using GLenum = unsigned int;
using GLuint = unsigned int;
using GLint = int;
using GLsizei = int;
using GLuint64 = std::uint64_t;
using GLsync = void*;

#if defined(_WIN32)
#define NOORRAY_GL_API __stdcall
#else
#define NOORRAY_GL_API
#endif

constexpr GLenum glNoError = 0;
constexpr GLenum glTexture2D = 0x0DE1;
constexpr GLenum glTextureWidth = 0x1000;
constexpr GLenum glTextureHeight = 0x1001;
constexpr GLenum glTextureInternalFormat = 0x1003;
constexpr GLenum glExtensions = 0x1F03;
constexpr GLenum glTextureBinding2D = 0x8069;
constexpr GLenum glTexture0 = 0x84C0;
constexpr GLenum glActiveTexture = 0x84E0;
constexpr GLenum glRgba16F = 0x881A;
constexpr GLenum glMaxCombinedTextureImageUnits = 0x8B4D;
constexpr GLenum glNumExtensions = 0x821D;
constexpr GLenum glSyncGpuCommandsComplete = 0x9117;
constexpr GLenum glTimeoutExpired = 0x911B;
constexpr GLenum glWaitFailed = 0x911D;
constexpr GLenum glSyncFlushCommandsBit = 0x1;
constexpr GLenum glTextureTiling = 0x9580;
constexpr GLenum glDedicatedMemoryObject = 0x9581;
constexpr GLenum glOptimalTiling = 0x9584;
#if defined(_WIN32)
constexpr GLenum glHandleTypeOpaqueWin32 = 0x9587;
#else
constexpr GLenum glHandleTypeOpaqueFd = 0x9586;
#endif
constexpr GLenum glLayoutGeneral = 0x958D;
constexpr GLenum glNumDeviceUuids = 0x9596;
constexpr GLenum glDeviceUuid = 0x9597;
constexpr GLuint64 copyTimeout = 1'000'000'000;

//! The OpenGL entry points of the interop; GL_EXT_memory_object and GL_EXT_semaphore are extensions, so all of them load from the current context.
struct Gl {
    GLenum(NOORRAY_GL_API* getError)();
    void(NOORRAY_GL_API* getIntegerv)(GLenum, GLint*);
    const unsigned char*(NOORRAY_GL_API* getStringi)(GLenum, GLuint);
    void(NOORRAY_GL_API* activeTexture)(GLenum);
    void(NOORRAY_GL_API* bindTexture)(GLenum, GLuint);
    void(NOORRAY_GL_API* genTextures)(GLsizei, GLuint*);
    void(NOORRAY_GL_API* deleteTextures)(GLsizei, const GLuint*);
    void(NOORRAY_GL_API* texParameteri)(GLenum, GLenum, GLint);
    void(NOORRAY_GL_API* getTexLevelParameteriv)(GLenum, GLint, GLenum, GLint*);
    void(NOORRAY_GL_API* copyImageSubData)(GLuint, GLenum, GLint, GLint, GLint, GLint, GLuint, GLenum, GLint, GLint, GLint, GLint, GLsizei, GLsizei, GLsizei);
    GLsync(NOORRAY_GL_API* fenceSync)(GLenum, unsigned int);
    GLenum(NOORRAY_GL_API* clientWaitSync)(GLsync, unsigned int, GLuint64);
    void(NOORRAY_GL_API* deleteSync)(GLsync);
    void(NOORRAY_GL_API* flush)();
    void(NOORRAY_GL_API* getUnsignedBytei_v)(GLenum, GLuint, unsigned char*);
    void(NOORRAY_GL_API* createMemoryObjects)(GLsizei, GLuint*);
    void(NOORRAY_GL_API* deleteMemoryObjects)(GLsizei, const GLuint*);
    void(NOORRAY_GL_API* memoryObjectParameteriv)(GLuint, GLenum, const GLint*);
    void(NOORRAY_GL_API* texStorageMem2D)(GLenum, GLsizei, GLenum, GLsizei, GLsizei, GLuint, GLuint64);
    void(NOORRAY_GL_API* genSemaphores)(GLsizei, GLuint*);
    void(NOORRAY_GL_API* deleteSemaphores)(GLsizei, const GLuint*);
    void(NOORRAY_GL_API* waitSemaphore)(GLuint, GLuint, const GLuint*, GLuint, const GLuint*, const GLenum*);
#if defined(_WIN32)
    void(NOORRAY_GL_API* importMemory)(GLuint, GLuint64, GLenum, void*);
    void(NOORRAY_GL_API* importSemaphore)(GLuint, GLenum, void*);
#else
    void(NOORRAY_GL_API* importMemory)(GLuint, GLuint64, GLenum, GLint);
    void(NOORRAY_GL_API* importSemaphore)(GLuint, GLenum, GLint);
#endif
};

using Procedure = void (*)();
using ProcedureLoader = Procedure (*)(const char*);

#if defined(_WIN32)
constexpr std::array<std::string_view, 4> requiredExtensions{"GL_EXT_memory_object", "GL_EXT_memory_object_win32", "GL_EXT_semaphore",
                                                             "GL_EXT_semaphore_win32"};

HMODULE openGlLibrary() {
    return GetModuleHandleA("opengl32.dll");
}

//! wglGetProcAddress gives only the functions after OpenGL 1.1, and opengl32.dll exports the rest.
Procedure loadProcedure(const char* name) {
    const HMODULE library = openGlLibrary();
    const auto wglGetProcAddress = reinterpret_cast<PROC(WINAPI*)(LPCSTR)>(GetProcAddress(library, "wglGetProcAddress"));
    PROC procedure = wglGetProcAddress(name);
    const auto value = reinterpret_cast<std::intptr_t>(procedure);
    if (value >= -1 && value <= 3)
        procedure = GetProcAddress(library, name);
    return reinterpret_cast<Procedure>(procedure);
}

std::expected<ProcedureLoader, Core::Error> currentLoader() {
    const HMODULE library = openGlLibrary();
    if (!library)
        return Core::fail(Core::ErrorCode::Unsupported, "the process has not loaded opengl32.dll");
    const auto wglGetCurrentContext = reinterpret_cast<HGLRC(WINAPI*)()>(GetProcAddress(library, "wglGetCurrentContext"));
    if (!wglGetCurrentContext || !wglGetCurrentContext())
        return Core::fail(Core::ErrorCode::InvalidState, "no OpenGL context is current");
    return loadProcedure;
}
#else
constexpr std::array<std::string_view, 4> requiredExtensions{"GL_EXT_memory_object", "GL_EXT_memory_object_fd", "GL_EXT_semaphore", "GL_EXT_semaphore_fd"};

ProcedureLoader platformLoader = nullptr;

//! GLX and EGL may both resolve a name, so a null result falls back to the symbols that the process has loaded.
Procedure loadProcedure(const char* name) {
    if (const Procedure procedure = platformLoader(name))
        return procedure;
    return reinterpret_cast<Procedure>(dlsym(RTLD_DEFAULT, name));
}

//! Blender makes its context with EGL on Wayland and with GLX on X11; the library with a current context gives the entry points.
std::expected<ProcedureLoader, Core::Error> currentLoader() {
    const std::array<std::array<const char*, 3>, 3> platforms{{
        {"libEGL.so.1", "eglGetCurrentContext", "eglGetProcAddress"},
        {"libGLX.so.0", "glXGetCurrentContext", "glXGetProcAddressARB"},
        {"libGL.so.1", "glXGetCurrentContext", "glXGetProcAddressARB"},
    }};
    for (const auto& [libraryName, currentName, loaderName] : platforms) {
        void* library = dlopen(libraryName, RTLD_LAZY | RTLD_NOLOAD);
        if (!library)
            continue;
        const auto current = reinterpret_cast<void* (*)()>(dlsym(library, currentName));
        const auto loader = reinterpret_cast<ProcedureLoader>(dlsym(library, loaderName));
        dlclose(library);
        if (!current || !loader || !current())
            continue;
        platformLoader = loader;
        return loadProcedure;
    }
    return Core::fail(Core::ErrorCode::InvalidState, "no OpenGL context is current in EGL or GLX");
}
#endif

template <class Function>
bool load(const ProcedureLoader loader, Function& function, const char* name, std::vector<std::string_view>& missing) {
    function = reinterpret_cast<Function>(loader(name));
    if (!function)
        missing.push_back(name);
    return function != nullptr;
}

std::optional<Gl> loaded;

std::expected<Gl*, Core::Error> loadGl() {
    if (loaded)
        return &*loaded;
    std::expected<ProcedureLoader, Core::Error> loader = currentLoader();
    if (!loader)
        return std::unexpected(std::move(loader.error()));
    Gl gl{};
    std::vector<std::string_view> missing;
    load(*loader, gl.getError, "glGetError", missing);
    load(*loader, gl.getIntegerv, "glGetIntegerv", missing);
    load(*loader, gl.getStringi, "glGetStringi", missing);
    load(*loader, gl.activeTexture, "glActiveTexture", missing);
    load(*loader, gl.bindTexture, "glBindTexture", missing);
    load(*loader, gl.genTextures, "glGenTextures", missing);
    load(*loader, gl.deleteTextures, "glDeleteTextures", missing);
    load(*loader, gl.texParameteri, "glTexParameteri", missing);
    load(*loader, gl.getTexLevelParameteriv, "glGetTexLevelParameteriv", missing);
    load(*loader, gl.copyImageSubData, "glCopyImageSubData", missing);
    load(*loader, gl.fenceSync, "glFenceSync", missing);
    load(*loader, gl.clientWaitSync, "glClientWaitSync", missing);
    load(*loader, gl.deleteSync, "glDeleteSync", missing);
    load(*loader, gl.flush, "glFlush", missing);
    load(*loader, gl.getUnsignedBytei_v, "glGetUnsignedBytei_vEXT", missing);
    load(*loader, gl.createMemoryObjects, "glCreateMemoryObjectsEXT", missing);
    load(*loader, gl.deleteMemoryObjects, "glDeleteMemoryObjectsEXT", missing);
    load(*loader, gl.memoryObjectParameteriv, "glMemoryObjectParameterivEXT", missing);
    load(*loader, gl.texStorageMem2D, "glTexStorageMem2DEXT", missing);
    load(*loader, gl.genSemaphores, "glGenSemaphoresEXT", missing);
    load(*loader, gl.deleteSemaphores, "glDeleteSemaphoresEXT", missing);
    load(*loader, gl.waitSemaphore, "glWaitSemaphoreEXT", missing);
#if defined(_WIN32)
    load(*loader, gl.importMemory, "glImportMemoryWin32HandleEXT", missing);
    load(*loader, gl.importSemaphore, "glImportSemaphoreWin32HandleEXT", missing);
#else
    load(*loader, gl.importMemory, "glImportMemoryFdEXT", missing);
    load(*loader, gl.importSemaphore, "glImportSemaphoreFdEXT", missing);
#endif
    if (!missing.empty())
        return Core::fail(Core::ErrorCode::Unsupported, std::format("the OpenGL context has no {}", missing.front()));
    GLint count = 0;
    gl.getIntegerv(glNumExtensions, &count);
    std::set<std::string_view> extensions;
    for (GLint index = 0; index < count; ++index)
        if (const unsigned char* name = gl.getStringi(glExtensions, static_cast<GLuint>(index)))
            extensions.insert(reinterpret_cast<const char*>(name));
    for (const std::string_view extension : requiredExtensions)
        if (!extensions.contains(extension))
            return Core::fail(Core::ErrorCode::Unsupported, std::format("the OpenGL context has no {}", extension));
    loaded = gl;
    return &*loaded;
}

//! OpenGL objects that a destructor gave up without a current context; the next call with one deletes them.
struct Garbage {
    GLuint memoryObject = 0;
    GLuint texture = 0;
    GLuint semaphore = 0;
    GLsync fence = nullptr;
};
std::vector<Garbage> garbage;

void collectGarbage(Gl& gl) {
    for (const Garbage& item : garbage) {
        if (item.fence) {
            gl.clientWaitSync(item.fence, glSyncFlushCommandsBit, copyTimeout);
            gl.deleteSync(item.fence);
        }
        if (item.semaphore)
            gl.deleteSemaphores(1, &item.semaphore);
        if (item.texture)
            gl.deleteTextures(1, &item.texture);
        if (item.memoryObject)
            gl.deleteMemoryObjects(1, &item.memoryObject);
    }
    garbage.clear();
}

//! Blender does not read glGetError in a release build, so older errors are cleared before a check of this code.
void clearErrors(Gl& gl) {
    for (int index = 0; index < 16 && gl.getError() != glNoError; ++index) {
    }
}

std::expected<void, Core::Error> checkGl(Gl& gl, const std::string_view operation) {
    const GLenum error = gl.getError();
    if (error == glNoError)
        return {};
    return Core::fail(Core::ErrorCode::InvalidState, std::format("OpenGL failed to {}: error 0x{:04X}", operation, error));
}

std::expected<void, Core::Error> checkDevice(Gl& gl, NoorRHI::Device& device) {
    auto handles = Core::check(NoorRHI::Interop::getDeviceHandles(device));
    if (!handles)
        return std::unexpected(std::move(handles.error()));
    GLint count = 0;
    gl.getIntegerv(glNumDeviceUuids, &count);
    for (GLint index = 0; index < count; ++index) {
        std::array<unsigned char, 16> uuid{};
        gl.getUnsignedBytei_v(glDeviceUuid, static_cast<GLuint>(index), uuid.data());
        if (std::ranges::equal(uuid, handles->deviceUuid))
            return {};
    }
    return Core::fail(Core::ErrorCode::Unsupported, "Blender's OpenGL context runs on another GPU than NoorRay");
}

//! Blender binds the texture of a GPUTexture to a unit when it draws with it, so the one bound RGBA16F texture of the film size is the target.
std::expected<GLuint, Core::Error> findTarget(Gl& gl, const std::set<GLuint>& own, const std::uint32_t width, const std::uint32_t height) {
    GLint units = 0;
    GLint active = 0;
    gl.getIntegerv(glMaxCombinedTextureImageUnits, &units);
    gl.getIntegerv(glActiveTexture, &active);
    std::set<GLuint> found;
    for (GLint unit = 0; unit < std::min(units, 96); ++unit) {
        gl.activeTexture(glTexture0 + static_cast<GLenum>(unit));
        GLint name = 0;
        gl.getIntegerv(glTextureBinding2D, &name);
        if (name == 0 || own.contains(static_cast<GLuint>(name)))
            continue;
        GLint textureWidth = 0;
        GLint textureHeight = 0;
        GLint format = 0;
        gl.getTexLevelParameteriv(glTexture2D, 0, glTextureWidth, &textureWidth);
        gl.getTexLevelParameteriv(glTexture2D, 0, glTextureHeight, &textureHeight);
        gl.getTexLevelParameteriv(glTexture2D, 0, glTextureInternalFormat, &format);
        if (static_cast<std::uint32_t>(textureWidth) == width && static_cast<std::uint32_t>(textureHeight) == height && format == GLint(glRgba16F))
            found.insert(static_cast<GLuint>(name));
    }
    gl.activeTexture(static_cast<GLenum>(active));
    if (found.size() != 1)
        return Core::fail(Core::ErrorCode::InvalidState,
                          std::format("{} bound RGBA16F textures have the size {}x{}; draw the target GPUTexture just before", found.size(), width, height));
    return *found.begin();
}

}

GlInterop::~GlInterop() {
    for (const Slot& slot : slots)
        garbage.push_back({slot.memoryObject, slot.texture, 0, static_cast<GLsync>(slot.fence)});
    garbage.push_back({0, 0, semaphore, nullptr});
}

void GlInterop::release() {
    for (Slot& slot : slots) {
        garbage.push_back({slot.memoryObject, slot.texture, 0, static_cast<GLsync>(slot.fence)});
        slot = {};
    }
    garbage.push_back({0, 0, semaphore, nullptr});
    semaphore = target = 0;
    signal.close();
    if (loaded)
        collectGarbage(*loaded);
}

namespace {

//! Imports the external memory of one image into a memory object and a texture of its own.
std::expected<void, Core::Error> importImage(Gl& gl, const NoorRHI::Image& image, GLuint& memoryObject, GLuint& texture) {
    auto memory = Core::check(NoorRHI::Interop::exportImageMemory(image));
    if (!memory)
        return std::unexpected(std::move(memory.error()));
    gl.createMemoryObjects(1, &memoryObject);
    const GLint dedicated = 1;
    gl.memoryObjectParameteriv(memoryObject, glDedicatedMemoryObject, &dedicated);
#if defined(_WIN32)
    //! OpenGL takes a reference of its own to the memory of an NT handle, so the handle closes after the import.
    gl.importMemory(memoryObject, memory->allocationSize, glHandleTypeOpaqueWin32, reinterpret_cast<void*>(memory->handle));
    CloseHandle(reinterpret_cast<HANDLE>(memory->handle));
    if (auto imported = checkGl(gl, "import the film memory"); !imported)
        return imported;
#else
    //! OpenGL owns the file descriptor after a successful import.
    gl.importMemory(memoryObject, memory->allocationSize, glHandleTypeOpaqueFd, static_cast<GLint>(memory->handle));
    if (auto imported = checkGl(gl, "import the film memory"); !imported) {
        close(static_cast<int>(memory->handle));
        return imported;
    }
#endif
    GLint bound = 0;
    gl.getIntegerv(glTextureBinding2D, &bound);
    gl.genTextures(1, &texture);
    gl.bindTexture(glTexture2D, texture);
    gl.texParameteri(glTexture2D, glTextureTiling, GLint(glOptimalTiling));
    gl.texStorageMem2D(glTexture2D, 1, glRgba16F, static_cast<GLsizei>(memory->width), static_cast<GLsizei>(memory->height), memoryObject, 0);
    gl.bindTexture(glTexture2D, static_cast<GLuint>(bound));
    return checkGl(gl, "make a texture of the film memory");
}

}

std::expected<void, Core::Error> GlInterop::open(NoorRHI::Device& device, const std::span<const NoorRHI::Image, slotCount> images) {
    std::expected<Gl*, Core::Error> loadedGl = loadGl();
    if (!loadedGl)
        return std::unexpected(std::move(loadedGl.error()));
    Gl& gl = **loadedGl;
    release();
    clearErrors(gl);
    if (auto same = checkDevice(gl, device); !same)
        return same;
    for (const NoorRHI::Image& image : images)
        if (image.getFormat() != NoorRHI::ImageFormat::Rgba16Float || image.getWidth() != images[0].getWidth() || image.getHeight() != images[0].getHeight())
            return Core::fail(Core::ErrorCode::InvalidArgument, "the shared images are not RGBA16F images of one size");
    width = images[0].getWidth();
    height = images[0].getHeight();
    std::set<GLuint> own;
    for (std::size_t index = 0; index < slotCount; ++index) {
        if (auto imported = importImage(gl, images[index], slots[index].memoryObject, slots[index].texture); !imported)
            return imported;
        own.insert(slots[index].texture);
    }
    auto handle = Core::check(signal.open(device));
    if (!handle)
        return std::unexpected(std::move(handle.error()));
    gl.genSemaphores(1, &semaphore);
#if defined(_WIN32)
    gl.importSemaphore(semaphore, glHandleTypeOpaqueWin32, reinterpret_cast<void*>(*handle));
    CloseHandle(reinterpret_cast<HANDLE>(*handle));
    if (auto imported = checkGl(gl, "import the Vulkan semaphore"); !imported)
        return imported;
#else
    gl.importSemaphore(semaphore, glHandleTypeOpaqueFd, static_cast<GLint>(*handle));
    if (auto imported = checkGl(gl, "import the Vulkan semaphore"); !imported) {
        close(static_cast<int>(*handle));
        return imported;
    }
#endif
    std::expected<GLuint, Core::Error> found = findTarget(gl, own, width, height);
    if (!found)
        return std::unexpected(std::move(found.error()));
    target = *found;
    return {};
}

std::expected<void, Core::Error> GlInterop::waitForCopy(const std::size_t slot) {
    if (slot >= slotCount)
        return Core::fail(Core::ErrorCode::InvalidArgument, "the slot is not in the OpenGL interop");
    if (!slots[slot].fence)
        return {};
    Gl& gl = *loaded;
    const auto fence = static_cast<GLsync>(slots[slot].fence);
    GLenum waited = glTimeoutExpired;
    for (int attempt = 0; attempt < 10 && waited == glTimeoutExpired; ++attempt)
        waited = gl.clientWaitSync(fence, glSyncFlushCommandsBit, copyTimeout);
    gl.deleteSync(fence);
    slots[slot].fence = nullptr;
    if (waited == glWaitFailed || waited == glTimeoutExpired)
        return Core::fail(Core::ErrorCode::InvalidState, "OpenGL did not finish the copy of the film");
    return {};
}

std::expected<void, Core::Error> GlInterop::present(const std::size_t slot) {
    if (!isOpen())
        return Core::fail(Core::ErrorCode::InvalidState, "the OpenGL interop is not open");
    if (slot >= slotCount)
        return Core::fail(Core::ErrorCode::InvalidArgument, "the slot is not in the OpenGL interop");
    Gl& gl = *loaded;
    collectGarbage(gl);
    if (auto waited = waitForCopy(slot); !waited)
        return waited;
    if (auto signalled = Core::check(signal.signal()); !signalled)
        return signalled;
    clearErrors(gl);
    //! NoorRHI keeps every image in the general layout.
    const GLenum layout = glLayoutGeneral;
    gl.waitSemaphore(semaphore, 0, nullptr, 1, &slots[slot].texture, &layout);
    gl.copyImageSubData(slots[slot].texture, glTexture2D, 0, 0, 0, 0, target, glTexture2D, 0, 0, 0, 0, static_cast<GLsizei>(width),
                        static_cast<GLsizei>(height), 1);
    slots[slot].fence = gl.fenceSync(glSyncGpuCommandsComplete, 0);
    //! The wait reaches the GPU before the next signal of the binary semaphore.
    gl.flush();
    return checkGl(gl, "copy the film");
}

}
