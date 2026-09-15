#include "MaterialXSceneRuntime.h"

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <exception>
#include <filesystem>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <MaterialXCore/Document.h>
#include <MaterialXCore/Types.h>
#include <MaterialXFormat/File.h>
#include <MaterialXFormat/XmlIo.h>

#include "Logging/Log.h"
#include "Materials/MaterialX/MaterialXDocument.h"
#include "Materials/SVM/SvmCompiler.h"
#include "Scene/Scene.h"


struct MaterialXSceneRuntime::Impl
{
    struct Job
    {
        std::size_t materialIndex{};
        std::uint64_t materialRevision{};
        MaterialX::DocumentPtr document;
        std::unordered_map<std::string, std::uint32_t> resolvedTextures;
    };

    struct Completion
    {
        std::size_t materialIndex{};
        std::uint64_t materialRevision{};
        std::optional<nr::svm::CompiledSvmProgram> result;
        std::string error;
    };

    mutable std::mutex mutex;
    std::condition_variable condition;
    std::deque<Job> jobs;
    std::deque<Completion> completed;
    std::unordered_set<std::size_t> scheduled;
    std::size_t active{};
    bool stopping{};
    std::thread worker;

    std::vector<std::pair<std::size_t, nr::svm::CompiledSvmProgram>> ready;
};

namespace {
std::string normalizedPath(const std::string& value)
{
    const std::filesystem::path path(value);
    std::error_code error;
    if (std::filesystem::exists(path))
        return std::filesystem::weakly_canonical(path, error).string();
    return path.lexically_normal().string();
}

std::unordered_map<std::string, std::uint32_t> resolveSceneTextures(
    const MaterialX::DocumentPtr& document, const Scene& scene,
    const std::string& sceneDirectory)
{
    std::unordered_map<std::string, std::uint32_t> resolved;
    for (const nr::materialx::MaterialXImageNode& image :
        nr::materialx::collectImageNodes(document)) {
        std::vector<std::string> candidates{image.rawFilePath};
        const std::filesystem::path rawPath(image.rawFilePath);
        if (!rawPath.is_absolute() && !sceneDirectory.empty())
            candidates.push_back((std::filesystem::path(sceneDirectory) / rawPath).string());

        for (std::size_t textureIndex = 0; textureIndex < scene.getTextures().size(); ++textureIndex) {
            const std::string texturePath = scene.getTextures()[textureIndex].getName();
            const std::string normalizedTexturePath = normalizedPath(texturePath);
            const bool matches = std::ranges::any_of(candidates,
                [&](const std::string& candidate) {
                    const std::string normalizedCandidate = normalizedPath(candidate);
                    const bool suffixMatch = normalizedTexturePath.size()
                        > normalizedCandidate.size()
                        && normalizedTexturePath.ends_with(normalizedCandidate)
                        && normalizedTexturePath[normalizedTexturePath.size()
                            - normalizedCandidate.size() - 1] == '/';
                    return candidate == texturePath
                        || normalizedCandidate == normalizedTexturePath
                        || suffixMatch;
                });
            if (matches) {
                resolved[image.rawFilePath] = static_cast<std::uint32_t>(textureIndex);
                break;
            }
        }
    }
    return resolved;
}
} // namespace

MaterialXSceneRuntime::MaterialXSceneRuntime()
    : impl_(std::make_unique<Impl>())
{
    impl_->worker = std::thread([this] {
        for (;;) {
            Impl::Job job;
            {
                std::unique_lock lock(impl_->mutex);
                impl_->condition.wait(lock, [this] {
                    return impl_->stopping || !impl_->jobs.empty();
                });
                if (impl_->stopping && impl_->jobs.empty())
                    return;
                job = std::move(impl_->jobs.front());
                impl_->jobs.pop_front();
                ++impl_->active;
            }

            Impl::Completion completion;
            completion.materialIndex = job.materialIndex;
            completion.materialRevision = job.materialRevision;
            try {
                nr::svm::SvmCompiler compiler;
                completion.result = compiler.compile(job.document, {},
                    job.resolvedTextures);
            } catch (const std::exception& error) {
                completion.error = error.what();
            } catch (...) {
                completion.error = "unknown exception";
            }

            {
                std::lock_guard lock(impl_->mutex);
                --impl_->active;
                impl_->scheduled.erase(job.materialIndex);
                impl_->completed.push_back(std::move(completion));
            }
            impl_->condition.notify_all();
        }
    });
}

MaterialXSceneRuntime::~MaterialXSceneRuntime()
{
    shutdown();
}

void MaterialXSceneRuntime::shutdown()
{
    {
        std::lock_guard lock(impl_->mutex);
        if (impl_->stopping)
            return;
        impl_->stopping = true;
    }
    impl_->condition.notify_all();
    if (impl_->worker.joinable())
        impl_->worker.join();
}

bool MaterialXSceneRuntime::needsCompilation(const Scene& scene) const
{
    return std::ranges::any_of(scene.getMaterials(),
        [](const Material& material) { return !material.hasProgram(); });
}

void MaterialXSceneRuntime::processPending(Scene& scene,
    const std::string& sceneDirectory)
{
    auto& materials = scene.getMaterials();
    const auto& paths = scene.getMaterialXSourcePaths();
    const auto& documents = scene.getMaterialXDocuments();
    std::deque<Impl::Completion> completed;
    {
        std::lock_guard lock(impl_->mutex);
        completed.swap(impl_->completed);
    }

    std::vector<std::size_t> fallbackCompiles;
    for (auto& completion : completed) {
        if (completion.materialRevision != scene.getMaterialRevision())
            continue;
        if (completion.result) {
            if (completion.materialIndex < materials.size()
                && !materials[completion.materialIndex].hasProgram())
                impl_->ready.emplace_back(completion.materialIndex,
                    std::move(*completion.result));
        } else {
            NR_LOG_WARN("MaterialX background compilation failed: "
                << completion.error);
            if (completion.materialIndex < materials.size()
                && !materials[completion.materialIndex].hasProgram())
                fallbackCompiles.push_back(completion.materialIndex);
        }
    }

    const auto schedule = [this](const std::size_t materialIndex,
                              const std::uint64_t materialRevision,
                              MaterialX::DocumentPtr document,
                              std::unordered_map<std::string, std::uint32_t> resolvedTextures) {
        {
            std::lock_guard lock(impl_->mutex);
            if (!impl_->scheduled.insert(materialIndex).second)
                return;
            impl_->jobs.push_back(Impl::Job{materialIndex, materialRevision,
                std::move(document), std::move(resolvedTextures)});
        }
        impl_->condition.notify_one();
    };

    for (const std::size_t materialIndex : fallbackCompiles) {
        NR_LOG_WARN("Falling back to the default MaterialX material for material "
            << materialIndex);
        schedule(materialIndex, scene.getMaterialRevision(),
            nr::materialx::defaultMaterial(), {});
    }

    for (std::size_t i = 0; i < materials.size(); ++i) {
        if (materials[i].hasProgram())
            continue;
        {
            std::lock_guard lock(impl_->mutex);
            if (impl_->scheduled.contains(i))
                continue;
        }
        if (std::ranges::any_of(impl_->ready,
            [i](const auto& ready) { return ready.first == i; }))
            continue;

        const bool hasSource = i < paths.size() && !paths[i].empty();
        const bool hasDocument = i < documents.size() && documents[i] != nullptr;
        MaterialX::DocumentPtr document;
        if (hasSource) {
            std::filesystem::path path(paths[i]);
            if (!path.is_absolute())
                path = std::filesystem::path(sceneDirectory) / path;
            path = path.lexically_normal();
            NR_LOG_INFO("Compiling MaterialX program: " << path.string());
            document = MaterialX::createDocument();
            MaterialX::FileSearchPath searchPath;
            searchPath.append(MaterialX::FilePath(path.parent_path().string()));
            try {
                MaterialX::readFromXmlFile(document, path.string(), searchPath);
            } catch (const std::exception& error) {
                NR_LOG_WARN("Failed to read MaterialX file " << path.string()
                    << ": " << error.what());
                document = nr::materialx::defaultMaterial();
            }
        } else if (hasDocument) {
            NR_LOG_INFO("Compiling in-memory MaterialX program for material " << i);
            document = documents[i]->copy();
        } else {
            NR_LOG_INFO("Compiling synthetic MaterialX program for native material " << i);
            document = nr::materialx::defaultMaterial();
        }
        auto resolvedTextures = document
            ? resolveSceneTextures(document, scene, sceneDirectory)
            : std::unordered_map<std::string, std::uint32_t>{};
        schedule(i, scene.getMaterialRevision(), std::move(document),
            std::move(resolvedTextures));
    }

    bool backgroundWork = false;
    {
        std::lock_guard lock(impl_->mutex);
        backgroundWork = !impl_->jobs.empty() || impl_->active != 0
            || !impl_->completed.empty();
    }
    if (backgroundWork || (needsCompilation(scene) && impl_->ready.empty()))
        return;

    for (auto& [materialIndex, program] : impl_->ready) {
        if (materialIndex >= materials.size() || materials[materialIndex].hasProgram())
            continue;
        // Publishing uploads this one material's buffers and marks the scene
        // dirty; no other material is re-uploaded.
        scene.setMaterialProgram(materialIndex, std::move(program));
    }
    impl_->ready.clear();
}

void MaterialXSceneRuntime::compileAndWait(Scene& scene,
    const std::string& sceneDirectory)
{
    processPending(scene, sceneDirectory);
    for (;;) {
        std::unique_lock lock(impl_->mutex);
        impl_->condition.wait(lock, [this] {
            return !impl_->completed.empty()
                || (impl_->jobs.empty() && impl_->active == 0);
        });
        lock.unlock();
        processPending(scene, sceneDirectory);
        if (!needsCompilation(scene))
            return;
    }
}
