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
        bool svmProgram{};
    };

    struct Completion
    {
        std::size_t materialIndex{};
        std::uint64_t materialRevision{};
        std::optional<nr::svm::CompiledSvmProgram> result;
        MaterialShaderProgram shaderProgram;
        std::string error;
    };

    struct Ready
    {
        std::size_t materialIndex{};
        nr::svm::CompiledSvmProgram program;
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

MaterialXSceneRuntime::MaterialXSceneRuntime()
    : impl_(std::make_unique<Impl>())
{
    // Both MaterialX's generator and a Slang session are worker-local. Four
    // workers keep large imports responsive without taking every CPU from the
    // render thread; the shared cache still compiles each shape only once.
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
                completion.result = nr::svm::CompiledSvmProgram{};
                if (job.svmProgram)
                    completion.result = nr::svm::SvmCompiler().compile(job.document, {},
                        job.resolvedTextures);
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
    return std::ranges::any_of(scene.getMaterials(),
        [](const Material& material) { return !material.compiled; });
}

void MaterialXSceneRuntime::processPending(Scene& scene, const bool svmPrograms,
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
                && !materials[completion.materialIndex].compiled)
                impl_->ready.push_back({completion.materialIndex,
                    std::move(*completion.result), std::move(completion.shaderProgram)});
        } else {
            NR_LOG_WARN("MaterialX background compilation failed: "
                << completion.error);
            if (completion.materialIndex < materials.size()
                && !materials[completion.materialIndex].compiled)
                fallbackCompiles.push_back(completion.materialIndex);
        }
    }

    const auto schedule = [this, svmPrograms](const std::size_t materialIndex,
                              const std::uint64_t materialRevision,
                              MaterialX::DocumentPtr document,
                              std::unordered_map<std::string, std::uint32_t> resolvedTextures) {
        {
            std::lock_guard lock(impl_->mutex);
            if (!impl_->scheduled.insert(materialIndex).second)
                return;
            impl_->jobs.push_back(Impl::Job{materialIndex, materialRevision,
                std::move(document), std::move(resolvedTextures), svmPrograms});
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
        if (materials[i].compiled)
            continue;
        {
            std::lock_guard lock(impl_->mutex);
            if (impl_->scheduled.contains(i))
                continue;
        }
        if (std::ranges::any_of(impl_->ready,
            [i](const Impl::Ready& ready) { return ready.materialIndex == i; }))
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
            NR_LOG_INFO("Compiling MaterialX material " << i);
            document = documents[i]->copy();
        } else {
            NR_LOG_INFO("Compiling default MaterialX material " << i);
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

    for (Impl::Ready& ready : impl_->ready) {
        if (ready.materialIndex >= materials.size() || materials[ready.materialIndex].compiled)
            continue;
        // Publishing uploads this one material's buffers and marks the scene
        // dirty; no other material is re-uploaded.
        scene.setMaterialProgram(ready.materialIndex, std::move(ready.program),
            std::move(ready.shaderProgram));
    }
    impl_->ready.clear();
}

void MaterialXSceneRuntime::compileAndWait(Scene& scene, const bool svmPrograms,
    const std::string& sceneDirectory)
{
    processPending(scene, svmPrograms, sceneDirectory);
    for (;;) {
        std::unique_lock lock(impl_->mutex);
        impl_->condition.wait(lock, [this] {
            return !impl_->completed.empty()
                || (impl_->jobs.empty() && impl_->active == 0);
        });
        lock.unlock();
        processPending(scene, svmPrograms, sceneDirectory);
        if (!needsCompilation(scene))
            return;
    }
}
