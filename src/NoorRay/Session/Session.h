#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <variant>

#include "NoorRHI/Commands/CommandList.h"
#include "NoorRHI/Resources/Readback.h"
#include "NoorRay/Core/Error.h"
#include "NoorRay/Gpu/GpuScene.h"
#include "NoorRay/Render/Renderer.h"
#include "NoorRay/Scene/Scene.h"

namespace NoorRay::Session {

using PickResult = std::variant<std::monostate, Scene::InstanceHandle, Scene::GaussianCloudHandle>;

//! The scene, its GPU data and the renderer on one device: the API of the app, the command line tool, Hydra and Python.
class Session {
public:
    //! A session can share the pipelines of another session on the device, as the material previews do.
    [[nodiscard]] std::expected<void, Core::Error> open(NoorRHI::Device& device, std::uint32_t width, std::uint32_t height,
                                                        std::shared_ptr<Render::PipelineCache> pipelines = nullptr);
    [[nodiscard]] std::expected<void, Core::Error> importFile(const std::filesystem::path& path);
    [[nodiscard]] std::expected<void, Core::Error> saveFile(const std::filesystem::path& path) const;
    [[nodiscard]] std::expected<void, Core::Error> resize(std::uint32_t width, std::uint32_t height);
    void clear();

    //! Decodes the image into the slot; colour slots read it as sRGB, the others as data. The slot's value is the factor of the texture.
    [[nodiscard]] std::expected<void, Core::Error> loadMaterialTexture(Scene::MaterialHandle material, Scene::TextureSlot slot,
                                                                       const std::filesystem::path& path);
    //! Empties the slot and removes the texture from the scene when nothing else uses it.
    void clearMaterialTexture(Scene::MaterialHandle material, Scene::TextureSlot slot);

    //! Applies the scene changes and records one frame into the renderer's display image.
    [[nodiscard]] std::expected<void, Core::Error> record(NoorRHI::CommandList& list);

    //! Copies the primary object id under the pixel; takePick gives the object once the GPU has written it.
    void requestPick(NoorRHI::CommandList& list, std::uint32_t x, std::uint32_t y);
    std::optional<PickResult> takePick();

    Scene::Scene& getScene() { return scene; }
    const Scene::Scene& getScene() const { return scene; }
    Render::Renderer& getRenderer() { return renderer; }
    const Render::Renderer& getRenderer() const { return renderer; }
    const Gpu::GpuScene& getGpuScene() const { return gpuScene; }
    NoorRHI::Device& getDevice() const { return *device; }

private:
    [[nodiscard]] std::expected<void, Core::Error> ensureCamera();

    NoorRHI::Device* device = nullptr;
    Scene::Scene scene;
    Gpu::GpuScene gpuScene;
    Render::Renderer renderer;
    NoorRHI::Readback<std::uint32_t> pickReadback;
    bool pickPending = false;
};

}
