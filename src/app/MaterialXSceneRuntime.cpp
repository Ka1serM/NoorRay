#include "MaterialXSceneRuntime.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <future>
#include <vector>

#include <MaterialXCore/Document.h>
#include <MaterialXCore/Types.h>
#include <MaterialXFormat/File.h>
#include <MaterialXFormat/XmlIo.h>

#include "Log.h"
#include "MaterialX/MaterialXCompiler.h"
#include "Raytracing/Runtime/Raytracer.h"
#include "Scene/Scene.h"
#include "SVM/SvmCompiler.h"

namespace mx = MaterialX;

struct MaterialXSceneRuntime::Impl
{
    struct PendingCompile
    {
        std::size_t materialIndex{};
        std::future<nr::svm::CompiledSvmProgram> future;
    };

    mx::DocumentPtr libraries = nr::materialx::getSharedStandardLibraries();
    std::vector<PendingCompile> pending;
    std::vector<std::pair<std::size_t, nr::svm::CompiledSvmProgram>> ready;
};

MaterialXSceneRuntime::MaterialXSceneRuntime()
    : impl_(std::make_unique<Impl>())
{
}

MaterialXSceneRuntime::~MaterialXSceneRuntime() = default;

bool MaterialXSceneRuntime::hasPendingCompilations() const
{
    return !impl_->pending.empty();
}

bool MaterialXSceneRuntime::needsCompilation(const Scene& scene) const
{
    if (!impl_->pending.empty())
        return false;
    for (uint32_t i = 0; i < scene.getMaterials().size(); ++i)
        if (scene.getMaterials()[i].svmBytecodeLength == 0)
            return true;
    return false;
}

void MaterialXSceneRuntime::compilePending(Scene& scene, Raytracer& raytracer,
    const std::string& sceneDirectory)
{
    auto& materials = scene.getMaterials();
    const auto& paths = scene.getMaterialXSourcePaths();
    const auto& documents = scene.getMaterialXDocuments();
    bool changed = false;

    const auto schedule = [this](const std::size_t materialIndex,
                              mx::DocumentPtr document) {
        document->importLibrary(impl_->libraries);
        impl_->pending.push_back(Impl::PendingCompile{
            materialIndex,
            std::async(std::launch::async,
                [document = std::move(document)]() {
                    nr::svm::SvmCompiler compiler;
                    return compiler.compile(document);
                })});
    };

    std::vector<std::size_t> fallbackCompiles;
    for (auto pending = impl_->pending.begin(); pending != impl_->pending.end();)
    {
        if (pending->future.wait_for(std::chrono::seconds(0))
            != std::future_status::ready)
        {
            ++pending;
            continue;
        }
        try {
            nr::svm::CompiledSvmProgram result = pending->future.get();
            if (pending->materialIndex < materials.size()
                && materials[pending->materialIndex].svmBytecodeLength == 0)
                impl_->ready.emplace_back(
                    pending->materialIndex, std::move(result));
        } catch (const std::exception& error) {
            LOG_WARN("MaterialX background compilation failed: "
                << error.what());
            // A graph that no longer compiles (for example because an
            // important node was deleted) must never strand the material on
            // the legacy default shader: fall back to a default MaterialX
            // material instead.
            if (pending->materialIndex < materials.size()
                && materials[pending->materialIndex].svmBytecodeLength == 0)
                fallbackCompiles.push_back(pending->materialIndex);
        }
        pending = impl_->pending.erase(pending);
    }
    for (const std::size_t materialIndex : fallbackCompiles)
    {
        LOG_WARN("Falling back to the default MaterialX material for material "
            << materialIndex);
        schedule(materialIndex, nr::materialx::defaultMaterial());
    }
    if (impl_->pending.empty() && !impl_->ready.empty())
    {
        for (auto& [materialIndex, program] : impl_->ready) {
            if (materialIndex >= materials.size()
                || materials[materialIndex].svmBytecodeLength != 0)
                continue;
            const nr::svm::SvmProgramRecord record =
                raytracer.registerMaterialXProgram(program);
            materials[materialIndex].svmBytecodeOffset = record.wordOffset;
            materials[materialIndex].svmBytecodeLength = record.wordCount;
            materials[materialIndex].svmTextureOffset = record.textureOffset;
            materials[materialIndex].svmTextureCount = record.textureCount;
            changed = true;
        }
        impl_->ready.clear();
    }

    for (std::size_t i = 0; i < materials.size(); ++i)
    {
        if (materials[i].svmBytecodeLength != 0)
            continue;
        if (std::ranges::any_of(impl_->pending,
            [i](const Impl::PendingCompile& pending) { return pending.materialIndex == i; }))
            continue;

        const bool hasSource = i < paths.size() && !paths[i].empty();
        const bool hasDocument = i < documents.size() && documents[i] != nullptr;
        mx::DocumentPtr document;
        if (hasSource) {
            std::filesystem::path path(paths[i]);
            if (!path.is_absolute())
                path = std::filesystem::path(sceneDirectory) / path;
            path = path.lexically_normal();
            LOG_INFO("Compiling MaterialX program: " << path.string());
            document = mx::createDocument();
            mx::FileSearchPath searchPath;
            searchPath.append(mx::FilePath(path.parent_path().string()));
            try {
                mx::readFromXmlFile(document, path.string(), searchPath);
            } catch (const std::exception& error) {
                LOG_WARN("Failed to read MaterialX file " << path.string()
                    << ": " << error.what());
                document = nr::materialx::defaultMaterial();
            }
        } else if (hasDocument) {
            LOG_INFO("Compiling in-memory MaterialX program for material " << i);
            // Clone so importLibrary() below never mutates the Scene's copy.
            document = documents[i]->copy();
        } else {
            LOG_INFO("Compiling synthetic MaterialX program for native material " << i);
            document = nr::materialx::defaultMaterial();
        }
        schedule(i, std::move(document));
    }
    if (changed)
    {
        scene.setDirtyFlag(Meshes);
        scene.setDirtyFlag(Accumulation);
    }
}
