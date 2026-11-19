#pragma once

#include <cstdint>
#include <deque>
#include <expected>
#include <functional>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include "NoorRHI/Commands/CommandList.h"
#include "NoorRHI/Resources/Image.h"
#include "NoorRay/Core/Error.h"
#include "NoorRay/Render/PipelineCache.h"
#include "NoorRay/Session/Session.h"

namespace NoorRay::Session {

//! Renders materials onto a sphere in a small NoorRay scene of its own and keeps each image, keyed by the material's contents, so an unchanged material renders
//! once.
class MaterialPreviews {
public:
    static constexpr std::uint32_t defaultSize = 128;
    static constexpr std::uint32_t defaultSamples = 48;
    static constexpr std::size_t defaultCapacity = 256;

    //! Sharing the pipelines of the main session means the previews compile no kernels of their own when the variants match.
    [[nodiscard]] std::expected<void, Core::Error> open(NoorRHI::Device& device, std::shared_ptr<Render::PipelineCache> pipelines,
                                                        std::uint32_t size = defaultSize, std::uint32_t samples = defaultSamples);

    //! The finished preview of a material of the source scene, or null while it waits; a missing preview joins the queue.
    const NoorRHI::Image* request(const Scene::Scene& source, Scene::MaterialHandle material);
    //! Records one frame of the preview in progress, starting the next queued one when needed; true while previews are left to render.
    [[nodiscard]] std::expected<bool, Core::Error> record(NoorRHI::CommandList& list, const Scene::Scene& source);
    bool isBusy() const { return job.has_value() || !queue.empty(); }
    std::uint32_t getSize() const { return size; }
    //! The radius of the sphere's outline over half the image width, so a view can cut the image to the sphere.
    float getSilhouette() const { return silhouette; }
    //! Changes with every finished preview, so a view knows when to look again.
    std::uint64_t getGeneration() const { return generation; }

    //! Called as each preview finishes, with the list that holds its last frame, so a tool can also download the renderer's film.
    using Finished = std::function<void(Scene::MaterialHandle material, NoorRHI::CommandList& list, const Render::Renderer& renderer)>;
    void setFinished(Finished callback) { finished = std::move(callback); }

    //! Identifies the material's look: its values and the contents of its textures, but not its name.
    static std::uint64_t hashMaterial(const Scene::Scene& scene, const Scene::Material& material);

private:
    struct Entry {
        NoorRHI::Image image;
        std::uint64_t lastUse = 0;
    };
    struct Job {
        std::uint64_t key = 0;
        Scene::MaterialHandle material;
        std::vector<Scene::TextureHandle> textures;
    };

    [[nodiscard]] std::expected<void, Core::Error> begin(const Scene::Scene& source);
    [[nodiscard]] std::expected<void, Core::Error> finish(NoorRHI::CommandList& list);
    void evict();

    NoorRHI::Device* device = nullptr;
    Session session;
    Scene::MaterialHandle sphereMaterial;
    std::uint32_t size = defaultSize;
    float silhouette = 1.0f;
    std::unordered_map<std::uint64_t, Entry> cache;
    std::deque<std::pair<std::uint64_t, Scene::MaterialHandle>> queue;
    std::optional<Job> job;
    std::uint64_t useCounter = 0;
    std::uint64_t generation = 0;
    Finished finished;
};

}
