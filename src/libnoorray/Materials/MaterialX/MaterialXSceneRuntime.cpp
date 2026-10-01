#include "MaterialXSceneRuntime.h"

#include <algorithm>
#include <chrono>
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
#include "Materials/MaterialX/SlangMaterialCompiler.h"
#include "Materials/MaterialX/SlangMaterialGenerator.h"
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
        std::optional<MaterialShaderProgram> shaderProgram;
        std::string error;
    };

    struct Ready
    {
        std::size_t materialIndex{};
        std::uint64_t materialRevision{};
        MaterialShaderProgram shaderProgram;
    };

    struct ShaderShapeEntry
    {
        // Retaining source makes the compact hash collision-safe without
        // hashing the generated module a second time.
        std::string source;
        std::shared_ptr<const nr::materialx::MaterialShader> shader;
        std::exception_ptr error;
        bool compiling{};
        std::condition_variable ready;
    };

    mutable std::mutex mutex;
    std::condition_variable condition;
    std::deque<Job> jobs;
    std::deque<Completion> completed;
    std::unordered_set<std::size_t> scheduled;
    std::unordered_map<std::uint64_t, std::vector<std::shared_ptr<ShaderShapeEntry>>> shaderShapes;
    std::size_t active{};
    bool stopping{};
    std::vector<std::thread> workers;
    std::function<void()> onCompleted;

    std::vector<Ready> ready;

    std::shared_ptr<const nr::materialx::MaterialShader> compileShaderShape(
        nr::materialx::SlangMaterialCompiler& compiler,
        const nr::materialx::SlangMaterial& material)
    {
        std::shared_ptr<ShaderShapeEntry> entry;
        bool owner = false;
        {
            std::unique_lock lock(mutex);
            auto& candidates = shaderShapes[material.shaderShape];
            const auto found = std::ranges::find_if(candidates, [&material](const auto& candidate) {
                return candidate->source == material.source;
            });
            if (found == candidates.end()) {
                entry = std::make_shared<ShaderShapeEntry>();
                entry->source = material.source;
                entry->compiling = true;
                candidates.push_back(entry);
                owner = true;
            } else {
                entry = *found;
                entry->ready.wait(lock, [&entry] { return !entry->compiling; });
            }
        }
        if (owner) {
            try {
                entry->shader = compiler.compile(material.source);
            } catch (...) {
                entry->error = std::current_exception();
            }
            {
                std::lock_guard lock(mutex);
                entry->compiling = false;
            }
            entry->ready.notify_all();
        }
        if (entry->error)
            std::rethrow_exception(entry->error);
        return entry->shader;
    }

    MaterialShaderProgram compileShaderProgram(
        const nr::materialx::SlangMaterialGenerator& generator,
        nr::materialx::SlangMaterialCompiler& compiler, const MaterialX::DocumentPtr& document,
        const std::unordered_map<std::string, std::uint32_t>& resolvedTextures);
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

struct SceneTextureLookup
{
    std::unordered_map<std::string, std::uint32_t> byPath;
    std::vector<std::pair<std::string, std::uint32_t>> normalizedPaths;
};

SceneTextureLookup makeSceneTextureLookup(const Scene& scene)
{
    SceneTextureLookup lookup;
    lookup.normalizedPaths.reserve(scene.getTextures().size());
    for (std::size_t index = 0; index < scene.getTextures().size(); ++index) {
        const std::string& path = scene.getTextures()[index].getName();
        const auto number = static_cast<std::uint32_t>(index);
        lookup.byPath.try_emplace(path, number);
        std::string normalized = normalizedPath(path);
        lookup.byPath.try_emplace(normalized, number);
        lookup.normalizedPaths.emplace_back(std::move(normalized), number);
    }
    return lookup;
}

std::unordered_map<std::string, std::uint32_t> resolveSceneTextures(
    const MaterialX::DocumentPtr& document, const SceneTextureLookup& lookup,
    const std::string& sceneDirectory)
{
    std::unordered_map<std::string, std::uint32_t> resolved;
    for (const nr::materialx::MaterialXImageNode& image :
        nr::materialx::collectImageNodes(document)) {
        std::vector<std::string> candidates{image.rawFilePath};
        const std::filesystem::path rawPath(image.rawFilePath);
        if (!rawPath.is_absolute() && !sceneDirectory.empty())
            candidates.push_back((std::filesystem::path(sceneDirectory) / rawPath).string());

        for (const std::string& candidate : candidates) {
            const std::string normalized = normalizedPath(candidate);
            if (const auto found = lookup.byPath.find(candidate); found != lookup.byPath.end()) {
                resolved[image.rawFilePath] = found->second;
                break;
            }
            if (const auto found = lookup.byPath.find(normalized); found != lookup.byPath.end()) {
                resolved[image.rawFilePath] = found->second;
                break;
            }
            // Preserve the previous suffix resolution for documents with relative image paths.
            for (const auto& [path, index] : lookup.normalizedPaths) {
                if (path.size() > normalized.size() && path.ends_with(normalized)
                    && path[path.size() - normalized.size() - 1] == '/') {
                    resolved[image.rawFilePath] = index;
                    break;
                }
            }
            if (resolved.contains(image.rawFilePath)) break;
        }
    }
    return resolved;
}
} // namespace

// Generates and compiles the realtime renderer's shader of a document. A
// document the realtime renderer cannot express is logged and left without
// a shader; the renderer shades it with its default surface.
MaterialShaderProgram MaterialXSceneRuntime::Impl::compileShaderProgram(
    const nr::materialx::SlangMaterialGenerator& generator,
    nr::materialx::SlangMaterialCompiler& compiler, const MaterialX::DocumentPtr& document,
    const std::unordered_map<std::string, std::uint32_t>& resolvedTextures)
{
    MaterialShaderProgram result;
    const auto started = std::chrono::steady_clock::now();
    try {
        nr::materialx::SlangMaterial material = generator.generate(document);
        const auto generated = std::chrono::steady_clock::now();
        result.shader = compileShaderShape(compiler, material);
        NR_LOG_INFO("Generated " << material.source.size() << " bytes of Slang in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(generated - started).count()
            << " ms and compiled it in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - generated).count() << " ms");
        result.parameters = std::move(material.parameters);
        result.transparent = material.transparent;
        for (const nr::materialx::SlangMaterialTexture& texture : material.textures) {
            const auto found = resolvedTextures.find(texture.file);
            result.textures.push_back({texture.word,
                found != resolvedTextures.end() ? found->second : ~0u});
        }
    } catch (const std::exception& error) {
        NR_LOG_WARN("MaterialX realtime shader generation failed: " << error.what());
        result = {};
    }
    return result;
}

MaterialXSceneRuntime::MaterialXSceneRuntime(std::function<void()> onCompleted)
    : impl_(std::make_unique<Impl>())
{
    impl_->onCompleted = std::move(onCompleted);
    // MaterialX generation and Slang compilation are heavy jobs, and each
    // compiled shader then feeds a driver compiler that also uses worker
    // threads. Let TBB own the wide import parallelism and keep this stage
    // bounded so it does not oversubscribe the machine. This also matches the
    // older import path that was substantially faster on high-core-count CPUs.
    const unsigned workerCount = std::clamp(std::thread::hardware_concurrency() > 1
        ? std::thread::hardware_concurrency() - 1 : 1u, 1u, 4u);
    impl_->workers.reserve(workerCount);
    for (unsigned worker = 0; worker < workerCount; ++worker)
        impl_->workers.emplace_back([this] {
        const nr::materialx::SlangMaterialGenerator generator;
        nr::materialx::SlangMaterialCompiler compiler;
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
                completion.shaderProgram = impl_->compileShaderProgram(generator, compiler,
                    job.document, job.resolvedTextures);
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
            if (impl_->onCompleted)
                impl_->onCompleted();
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
    for (std::thread& worker : impl_->workers)
        if (worker.joinable())
            worker.join();
}

bool MaterialXSceneRuntime::needsCompilation(const Scene& scene) const
{
    return std::ranges::any_of(scene.getMaterials(), [](const Material& material) {
        return !material.compiled && material.kind == MaterialKind::Surface;
    });
}

bool MaterialXSceneRuntime::processPending(Scene& scene, const std::string& sceneDirectory)
{
    auto& materials = scene.getMaterials();
    const auto& paths = scene.getMaterialXSourcePaths();
    const auto& documents = scene.getMaterialXDocuments();
    std::deque<Impl::Completion> completed;
    {
        std::lock_guard lock(impl_->mutex);
        completed.swap(impl_->completed);
    }

    // A program compiled from a document the material no longer holds, or
    // for a material a clear() replaced, must not be published.
    const auto current = [&materials](const std::size_t index, const std::uint64_t revision) {
        return index < materials.size() && materials[index].revision == revision;
    };
    std::erase_if(impl_->ready, [&current](const Impl::Ready& ready) {
        return !current(ready.materialIndex, ready.materialRevision);
    });

    std::vector<std::size_t> fallbackCompiles;
    // Materials edited while they compiled; their stale result is dropped.
    std::vector<std::size_t> recompiles;
    for (auto& completion : completed) {
        if (!current(completion.materialIndex, completion.materialRevision)) {
            if (completion.materialIndex < materials.size()
                && !materials[completion.materialIndex].compiled)
                recompiles.push_back(completion.materialIndex);
            continue;
        }
        if (completion.shaderProgram) {
            if (completion.materialIndex < materials.size()
                && !materials[completion.materialIndex].compiled)
                impl_->ready.push_back({completion.materialIndex, completion.materialRevision,
                    std::move(*completion.shaderProgram)});
        } else {
            NR_LOG_WARN("MaterialX background compilation failed: "
                << completion.error);
            if (completion.materialIndex < materials.size()
                && !materials[completion.materialIndex].compiled)
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
        schedule(materialIndex, materials[materialIndex].revision,
            nr::materialx::defaultMaterial(), {});
    }

    // Only materials the scene listed, and those whose compile went stale,
    // are considered; the rest are compiled or already on their way.
    std::vector<std::size_t> candidates(recompiles);
    for (const uint32_t index : scene.takeMaterialsToCompile())
        candidates.push_back(index);
    std::unordered_set<std::size_t> readyIndices;
    for (const Impl::Ready& ready : impl_->ready)
        readyIndices.insert(ready.materialIndex);
    const SceneTextureLookup textureLookup = makeSceneTextureLookup(scene);
    for (const std::size_t i : candidates) {
        // Splat materials are drawn by built-in stages and have nothing to compile.
        if (i >= materials.size() || materials[i].compiled || readyIndices.contains(i)
            || materials[i].kind != MaterialKind::Surface)
            continue;
        {
            std::lock_guard lock(impl_->mutex);
            if (impl_->scheduled.contains(i))
                continue;
        }

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
            NR_LOG_INFO("Compiling MaterialX material " << i);
            document = documents[i]->copy();
        } else {
            NR_LOG_INFO("Compiling default MaterialX material " << i);
            document = nr::materialx::defaultMaterial();
        }
        auto resolvedTextures = document
            ? resolveSceneTextures(document, textureLookup, sceneDirectory)
            : std::unordered_map<std::string, std::uint32_t>{};
        schedule(i, materials[i].revision, std::move(document),
            std::move(resolvedTextures));
    }

    bool backgroundWork = false;
    {
        std::lock_guard lock(impl_->mutex);
        backgroundWork = !impl_->jobs.empty() || impl_->active != 0
            || !impl_->completed.empty();
    }
    // Feed the driver a steady stream of small shader waves while imports are
    // still running. Waiting for every MaterialX job to finish produces one
    // very large ray-tracing compilation burst at the end of a level batch.
    constexpr std::size_t publicationWave = 32;
    if (impl_->ready.empty() || (backgroundWork && impl_->ready.size() < publicationWave))
        return false;

    bool published = false;
    const std::size_t count = std::min(publicationWave, impl_->ready.size());
    for (std::size_t index = 0; index < count; ++index) {
        Impl::Ready& ready = impl_->ready[index];
        if (ready.materialIndex >= materials.size() || materials[ready.materialIndex].compiled)
            continue;
        // Publishing uploads this one material's buffers and marks the scene
        // dirty; no other material is re-uploaded.
        scene.setMaterialProgram(ready.materialIndex, std::move(ready.shaderProgram));
        published = true;
    }
    impl_->ready.erase(impl_->ready.begin(), impl_->ready.begin() + count);
    return published;
}

void MaterialXSceneRuntime::compileAndWait(Scene& scene, const std::string& sceneDirectory)
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
