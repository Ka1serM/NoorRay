#include "renderParam.h"

#include "imageLoader.h"
#include "material.h"
#include "memoryTextureRegistry.h"

#include "Materials/MaterialX/MaterialXDocument.h"

#include <MaterialXCore/Document.h>

#include <pxr/base/arch/hash.h>
#include <pxr/base/tf/diagnostic.h>
#include <pxr/usd/ar/asset.h>
#include <pxr/usd/ar/resolver.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdio>
#include <exception>
#include <future>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "Backend/Vulkan/Raytracer/RaytracerRenderer.h"
#include "Scene/Resources/Texture.h"

namespace mx = MaterialX;

PXR_NAMESPACE_OPEN_SCOPE

namespace
{
template<typename T>
std::vector<T> FlipImageRows(const std::vector<T>& source,
    const int width, const int height)
{
    std::vector<T> result(source.size());
    const size_t rowSize = static_cast<size_t>(width) * 4;
    for (int y = 0; y < height; ++y) {
        const auto sourceRow = source.begin()
            + static_cast<size_t>(height - 1 - y) * rowSize;
        std::copy_n(sourceRow, rowSize,
            result.begin() + static_cast<size_t>(y) * rowSize);
    }
    return result;
}

uint64_t HashMaterialXDocument(const std::string_view xml)
{
    // FNV-1a is deterministic, fast for the small/medium XML payloads passed
    // through Hydra, and does not depend on a platform's std::hash seed.
    uint64_t hash = 14695981039346656037ull;
    for (const unsigned char byte : xml) {
        hash ^= byte;
        hash *= 1099511628211ull;
    }
    return hash;
}

std::string AssetFlightKey(const std::string& resolvedPath,
    const uint64_t size, const bool timestampValid,
    const uint64_t timestampBits)
{
    std::string key;
    key.reserve(resolvedPath.size() + 48);
    key = resolvedPath;
    key.push_back('\n');
    key += std::to_string(size);
    key.push_back('\n');
    key += timestampValid ? std::to_string(timestampBits) : "invalid";
    return key;
}

} // namespace

HdNoorRayRenderParam::HdNoorRayRenderParam()
{
}

HdNoorRayRenderParam::~HdNoorRayRenderParam() noexcept
{
    try {
        materialCompileTasks_.wait();
    } catch (const std::exception& error) {
        fprintf(stderr,
            "[hdNoorRay] a background material compile failed during "
            "shutdown: %s\n", error.what());
    } catch (...) {
        fprintf(stderr,
            "[hdNoorRay] a background material compile failed during "
            "shutdown: unknown exception\n");
    }
    if (session.raytracer)
        session.raytracer->device().synchronize();
    // Give up every reference before the session (and with it the registries
    // they point into) is destroyed.
    textureCache_.clear();
    decodedTextureCache_.clear();
    decodedTextureFlights_.clear();
    assetHashFlights_.clear();
    assetFingerprintCache_.clear();
    materialBindings_.clear();
    materials_.clear();
    if (session.raytracer)
        session.raytracer.reset();
}

HdNoorRayRenderParam::ContentIdentity
HdNoorRayRenderParam::GetTextureContentIdentity(
    const std::string& filePath)
{
    ArResolver& resolver = ArGetResolver();
    const ArResolvedPath resolvedPath = resolver.Resolve(filePath);
    if (resolvedPath.empty())
        throw std::runtime_error(
            "the asset resolver could not resolve the image path");

    const std::shared_ptr<ArAsset> asset =
        resolver.OpenAsset(resolvedPath);
    if (!asset)
        throw std::runtime_error(
            "the asset resolver could not open the image");

    const uint64_t size = asset->GetSize();
    const ArTimestamp timestamp =
        resolver.GetModificationTimestamp(filePath, resolvedPath);
    const bool timestampValid = timestamp.IsValid();
    const uint64_t timestampBits = timestampValid
        ? std::bit_cast<uint64_t>(timestamp.GetTime())
        : 0;
    const std::string& resolved = resolvedPath.GetPathString();

    if (timestampValid) {
        std::scoped_lock lock(mutex);
        const auto cached = assetFingerprintCache_.find(resolved);
        if (cached != assetFingerprintCache_.end()
            && cached->second.size == size
            && cached->second.timestampBits == timestampBits) {
            return cached->second.content;
        }
    }

    const std::string flightKey =
        AssetFlightKey(resolved, size, timestampValid, timestampBits);
    std::shared_future<ContentIdentity> future;
    std::shared_ptr<std::promise<ContentIdentity>> promise;
    bool ownsHash = false;
    {
        std::scoped_lock lock(mutex);
        const auto inFlight = assetHashFlights_.find(flightKey);
        if (inFlight != assetHashFlights_.end()) {
            future = inFlight->second;
        } else {
            promise = std::make_shared<std::promise<ContentIdentity>>();
            future = promise->get_future().share();
            assetHashFlights_.emplace(flightKey, future);
            ownsHash = true;
        }
    }
    if (!ownsHash)
        return future.get();

    try {
        constexpr size_t chunkSize = 1024 * 1024;
        std::array<char, chunkSize> bytes;
        uint64_t first = 0x243f6a8885a308d3ull;
        uint64_t second = 0x13198a2e03707344ull;
        for (uint64_t offset = 0; offset < size;) {
            const size_t count = static_cast<size_t>(
                std::min<uint64_t>(bytes.size(), size - offset));
            if (asset->Read(bytes.data(), count, offset) != count)
                throw std::runtime_error(
                    "the asset resolver could not read the complete image");
            first = ArchHash64(bytes.data(), count, first);
            second = ArchHash64(bytes.data(), count, second);
            offset += count;
        }
        const ContentIdentity content{first, second, size};
        {
            std::scoped_lock lock(mutex);
            if (timestampValid) {
                assetFingerprintCache_.insert_or_assign(resolved,
                    AssetFingerprint{size, timestampBits, content});
            }
            assetHashFlights_.erase(flightKey);
        }
        promise->set_value(content);
        return content;
    } catch (...) {
        {
            std::scoped_lock lock(mutex);
            assetHashFlights_.erase(flightKey);
        }
        promise->set_exception(std::current_exception());
        throw;
    }
}

HdNoorRayRenderParam::DecodedTextureData
HdNoorRayRenderParam::GetOrDecodeTexture(
    const std::string& filePath, const ContentIdentity& content,
    const bool flipY)
{
    const DecodedTextureCacheKey decodeKey{content, flipY};
    std::shared_future<DecodedTextureData> future;
    std::shared_ptr<std::promise<DecodedTextureData>> promise;
    bool ownsDecode = false;
    {
        std::scoped_lock lock(mutex);
        const auto cached = decodedTextureCache_.find(decodeKey);
        if (cached != decodedTextureCache_.end()) {
            DecodedTextureData decoded;
            decoded.width = cached->second.width;
            decoded.height = cached->second.height;
            decoded.pixelType = cached->second.pixelType;
            switch (decoded.pixelType) {
            case DecodedPixelType::Rgba8:
                decoded.rgba8 = cached->second.rgba8.lock();
                if (decoded.rgba8)
                    return decoded;
                break;
            case DecodedPixelType::Rgba16Float:
                decoded.rgba16Float =
                    cached->second.rgba16Float.lock();
                if (decoded.rgba16Float)
                    return decoded;
                break;
            case DecodedPixelType::Rgba32Float:
                decoded.rgba32Float =
                    cached->second.rgba32Float.lock();
                if (decoded.rgba32Float)
                    return decoded;
                break;
            }
            decodedTextureCache_.erase(cached);
        }

        const auto inFlight = decodedTextureFlights_.find(decodeKey);
        if (inFlight != decodedTextureFlights_.end()) {
            future = inFlight->second;
        } else {
            promise = std::make_shared<std::promise<DecodedTextureData>>();
            future = promise->get_future().share();
            decodedTextureFlights_.emplace(decodeKey, future);
            ownsDecode = true;
        }
    }
    if (!ownsDecode)
        return future.get();

    try {
        DecodedTextureData decoded;
        HdNoorRayDecodedImage image;
        std::string hioError;
        if (HdNoorRayLoadImage(filePath, &image, &hioError, flipY)) {
            decoded.width = image.width;
            decoded.height = image.height;
            switch (image.pixelType) {
            case HdNoorRayImagePixelType::Rgba8:
                decoded.pixelType = DecodedPixelType::Rgba8;
                decoded.rgba8 =
                    std::make_shared<const std::vector<uint8_t>>(
                        std::move(image.rgba8));
                break;
            case HdNoorRayImagePixelType::Rgba16Float:
                decoded.pixelType = DecodedPixelType::Rgba16Float;
                decoded.rgba16Float =
                    std::make_shared<const std::vector<uint16_t>>(
                        std::move(image.rgba16Float));
                break;
            case HdNoorRayImagePixelType::Rgba32Float:
                decoded.pixelType = DecodedPixelType::Rgba32Float;
                decoded.rgba32Float =
                    std::make_shared<const std::vector<float>>(
                        std::move(image.rgba32Float));
                break;
            }
        } else {
            // Retain the original decoder for image plugins Hio does not
            // provide. Its shared backing storage is adopted into the same
            // content cache as an Hio result.
            const Texture fallback(filePath, TextureEncoding::Linear8);
            decoded.width = fallback.getWidth();
            decoded.height = fallback.getHeight();
            if (fallback.usesByteStorage()) {
                decoded.pixelType = DecodedPixelType::Rgba8;
                decoded.rgba8 = flipY
                    ? std::make_shared<const std::vector<uint8_t>>(
                        FlipImageRows(fallback.getBytePixels(),
                            decoded.width, decoded.height))
                    : fallback.getByteStorage();
            } else if (fallback.usesHalfStorage()) {
                decoded.pixelType = DecodedPixelType::Rgba16Float;
                decoded.rgba16Float = flipY
                    ? std::make_shared<const std::vector<uint16_t>>(
                        FlipImageRows(fallback.getHalfPixels(),
                            decoded.width, decoded.height))
                    : fallback.getHalfStorage();
            } else {
                decoded.pixelType = DecodedPixelType::Rgba32Float;
                decoded.rgba32Float = flipY
                    ? std::make_shared<const std::vector<float>>(
                        FlipImageRows(fallback.getPixels(),
                            decoded.width, decoded.height))
                    : fallback.getFloatStorage();
            }
        }

        DecodedTextureCacheEntry cacheEntry;
        cacheEntry.width = decoded.width;
        cacheEntry.height = decoded.height;
        cacheEntry.pixelType = decoded.pixelType;
        cacheEntry.rgba8 = decoded.rgba8;
        cacheEntry.rgba16Float = decoded.rgba16Float;
        cacheEntry.rgba32Float = decoded.rgba32Float;
        {
            std::scoped_lock lock(mutex);
            decodedTextureCache_.insert_or_assign(
                decodeKey, std::move(cacheEntry));
            decodedTextureFlights_.erase(decodeKey);
        }
        promise->set_value(decoded);
        return decoded;
    } catch (...) {
        {
            std::scoped_lock lock(mutex);
            decodedTextureFlights_.erase(decodeKey);
        }
        promise->set_exception(std::current_exception());
        throw;
    }
}

TextureHandle HdNoorRayRenderParam::GetOrCreateTexture(
    const std::string& filePath, const TextureEncoding encoding,
    const bool flipY)
{
    try {
        if (filePath.starts_with("noorray-memory://"))
            return GetOrCreateMemoryTexture(filePath, encoding);
        const ContentIdentity content =
            GetTextureContentIdentity(filePath);
        const DecodedTextureData decoded =
            GetOrDecodeTexture(filePath, content, flipY);
        const TextureEncoding effectiveEncoding =
            decoded.pixelType == DecodedPixelType::Rgba16Float
            ? TextureEncoding::Float16
            : decoded.pixelType == DecodedPixelType::Rgba32Float
            ? TextureEncoding::Float32
            : encoding;
        const TextureCacheKey cacheKey{content, effectiveEncoding, flipY};

        {
            std::scoped_lock lock(mutex);
            const auto existing = textureCache_.find(cacheKey);
            if (existing != textureCache_.end()) {
                if (session.scene.getTexture(existing->second))
                    return existing->second;
                textureCache_.erase(existing);
            }
        }

        std::optional<Texture> textureData;
        switch (decoded.pixelType) {
        case DecodedPixelType::Rgba8:
            textureData.emplace(filePath, decoded.rgba8,
                decoded.width, decoded.height, effectiveEncoding);
            break;
        case DecodedPixelType::Rgba16Float:
            textureData.emplace(filePath, decoded.rgba16Float,
                decoded.width, decoded.height, effectiveEncoding);
            break;
        case DecodedPixelType::Rgba32Float:
            textureData.emplace(filePath, decoded.rgba32Float,
                decoded.width, decoded.height, effectiveEncoding);
            break;
        }

        std::scoped_lock lock(mutex);
        const auto existing = textureCache_.find(cacheKey);
        if (existing != textureCache_.end()) {
            if (session.scene.getTexture(existing->second))
                return existing->second;
            textureCache_.erase(existing);
        }
        const TextureHandle texture = session.scene.addTexture(std::move(*textureData));
        textureCache_.insert_or_assign(cacheKey, texture);
        return texture;
    } catch (const std::exception& error) {
        TF_WARN("hdNoorRay could not load texture '%s': %s",
            filePath.c_str(), error.what());
        return {};
    }
}

TextureHandle HdNoorRayRenderParam::GetOrCreateMemoryTexture(
    const std::string& uri, const TextureEncoding encoding)
{
    (void) encoding;
    const std::shared_ptr<const hdnoorray::MemoryTexturePixels> image =
        hdnoorray::findMemoryTexture(uri);
    if (!image || !image->rgba)
        throw std::runtime_error("the Blender memory texture is no longer registered");
    const std::vector<float>& pixels = *image->rgba;

    const ContentIdentity content{
        ArchHash64(reinterpret_cast<const char*>(pixels.data()), pixels.size() * sizeof(float),
            0x243f6a8885a308d3ull),
        ArchHash64(reinterpret_cast<const char*>(pixels.data()), pixels.size() * sizeof(float),
            0x13198a2e03707344ull),
        static_cast<uint64_t>(pixels.size() * sizeof(float))};
    // Float memory data is already Blender's scene-linear image buffer.
    // Its requested MaterialX color space is intentionally not reapplied.
    const TextureCacheKey cacheKey{content, TextureEncoding::Float32};
    {
        std::scoped_lock lock(mutex);
        const auto existing = textureCache_.find(cacheKey);
        if (existing != textureCache_.end()) {
            if (session.scene.getTexture(existing->second))
                return existing->second;
            textureCache_.erase(existing);
        }
    }

    Texture texture(uri, image->rgba, image->width, image->height,
        TextureEncoding::Float32);
    std::scoped_lock lock(mutex);
    const auto existing = textureCache_.find(cacheKey);
    if (existing != textureCache_.end()) {
        if (session.scene.getTexture(existing->second))
            return existing->second;
        textureCache_.erase(existing);
    }
    const TextureHandle result = session.scene.addTexture(std::move(texture));
    textureCache_.insert_or_assign(cacheKey, result);
    return result;
}

void HdNoorRayRenderParam::PruneTextureCache()
{
    std::scoped_lock lock(mutex);
    std::erase_if(textureCache_, [this](const auto& entry) {
        return session.scene.getTexture(entry.second) == nullptr;
    });
    std::erase_if(decodedTextureCache_, [](const auto& entry) {
        switch (entry.second.pixelType) {
        case DecodedPixelType::Rgba8:
            return entry.second.rgba8.expired();
        case DecodedPixelType::Rgba16Float:
            return entry.second.rgba16Float.expired();
        case DecodedPixelType::Rgba32Float:
            return entry.second.rgba32Float.expired();
        }
        return true;
    });
}

void HdNoorRayRenderParam::QueueMaterialCompilation(
    const SdfPath& id, MaterialX::DocumentPtr document,
    std::function<MaterialCompilationOutput()> compile)
{
    uint64_t generation;
    {
        std::scoped_lock lock(mutex);
        generation = ++materialCompileGenerations_[id];
        // Keep the currently published material while an edit compiles. A new
        // prim gets its document published immediately; the completed compile
        // then fills that same material slot's program in one render-thread
        // operation.
        const auto existing = materials_.find(id);
        if (existing == materials_.end() || !existing->second.isValid()) {
            PublishMaterial(id, document);
        }
    }
    pendingMaterialCompiles_.fetch_add(1, std::memory_order_relaxed);
    materialCompileTasks_.run(
        [this, id, generation, document, compile = std::move(compile)]() {
            MaterialCompilationResult result{
                id, generation, document, {}, {}};
            {
                // Scene imports can replace a material several times while
                // older jobs are still sitting in TBB's queue. Discard a
                // superseded job before it acquires a compiler and starts the
                // MaterialX-to-SVM graph compilation.
                std::scoped_lock lock(mutex);
                const auto current = materialCompileGenerations_.find(id);
                if (current == materialCompileGenerations_.end()
                    || current->second != generation) {
                    completedMaterialCompiles_.push(std::move(result));
                    return;
                }
            }
            try {
                result.output = compile();
            } catch (const std::exception& error) {
                result.error = error.what();
            } catch (...) {
                result.error = "unknown MaterialX compilation failure";
            }
            completedMaterialCompiles_.push(std::move(result));
        });
}

bool HdNoorRayRenderParam::ProcessMaterialCompilations()
{
    bool materialsChanged = false;
    MaterialCompilationResult result;
    while (completedMaterialCompiles_.try_pop(result)) {
        pendingMaterialCompiles_.fetch_sub(1, std::memory_order_relaxed);
        readyMaterialCompiles_.push_back(std::move(result));
    }
    // A scene import can enqueue hundreds of materials. Registering each
    // Commit a completed wave together so libnoorray uploads one immutable
    // Vulkan material table rather than replacing it for each material.
    if (pendingMaterialCompiles_.load(std::memory_order_relaxed) != 0)
        return false;

    // libnoorray owns the material program table now. Hydra's worker jobs
    // remain useful for validating documents, but publication invalidates the
    // scene slot and lets MaterialXSceneRuntime compile/upload one immutable
    // Vulkan table for the completed batch.
    for (MaterialCompilationResult& result : readyMaterialCompiles_) {
        std::scoped_lock lock(mutex);
        const auto generation = materialCompileGenerations_.find(result.id);
        if (generation == materialCompileGenerations_.end()
            || generation->second != result.generation)
            continue;
        if (!result.error.empty()) {
            TF_WARN("hdNoorRay: MaterialX compile failed for %s: %s",
                result.id.GetText(), result.error.c_str());
            // A graph that no longer compiles (for example because an
            // important node was deleted) must never leave the material broken
            // or stuck on a stale last-good program: fall back to the default
            // MaterialX material. The synthetic default is a tiny graph that
            // cannot fail, so compiling it here is safe.
            try {
                mx::DocumentPtr fallbackDocument =
                    nr::materialx::defaultMaterial();
                fallbackDocument->setDataLibrary(
                    nr::materialx::getSharedStandardLibraries());
                result.document = std::move(fallbackDocument);
                result.error.clear();
                PublishMaterial(result.id, result.document);
                session.scene.invalidateMaterial(materials_[result.id].handle());
                materialsChanged = true;
            } catch (const std::exception& error) {
                TF_WARN(
                    "hdNoorRay: default MaterialX fallback also failed for %s: %s",
                    result.id.GetText(), error.what());
            }
            continue;
        }
        PublishMaterial(result.id, result.document);
        session.scene.invalidateMaterial(materials_[result.id].handle());
        materialsChanged = true;
    }
    readyMaterialCompiles_.clear();
    if (materialsChanged && session.raytracer)
        session.rebuildNativeMaterials();
    return materialsChanged;
}

void HdNoorRayRenderParam::SetProgress(const double progress)
{
    progress_.store(std::clamp(progress, 0.0, 1.0), std::memory_order_relaxed);
}

double HdNoorRayRenderParam::GetProgress() const
{
    return progress_.load(std::memory_order_relaxed);
}

void HdNoorRayRenderParam::AccumulateGpuTimeMs(const float ms)
{
    cumulativeGpuTimeSeconds_.fetch_add(ms / 1000.0, std::memory_order_relaxed);
}

double HdNoorRayRenderParam::GetTotalClockTime() const
{
    return cumulativeGpuTimeSeconds_.load(std::memory_order_relaxed);
}

void HdNoorRayRenderParam::ResetClock()
{
    cumulativeGpuTimeSeconds_.store(0.0, std::memory_order_relaxed);
    progress_.store(0.0, std::memory_order_relaxed);
}

bool HdNoorRayRenderParam::SetMaterialXDocument(
    std::string materialLeaf, std::string xml)
{
    const uint64_t hash = HashMaterialXDocument(xml);
    std::unique_lock lock(materialXDocumentsMutex_);
    const auto existing = materialXDocuments_.find(materialLeaf);
    if (existing != materialXDocuments_.end()
        && existing->second.hash == hash
        && existing->second.xml
        && existing->second.xml->size() == xml.size()
        && *existing->second.xml == xml) {
        return false;
    }

    MaterialXDocument document;
    document.xml = std::make_shared<const std::string>(std::move(xml));
    document.hash = hash;
    document.revision = ++nextMaterialXDocumentRevision_;
    materialXDocuments_.insert_or_assign(
        std::move(materialLeaf), std::move(document));
    return true;
}

std::optional<HdNoorRayRenderParam::MaterialXDocument>
HdNoorRayRenderParam::GetMaterialXDocument(
    const std::string& materialLeaf) const
{
    std::shared_lock lock(materialXDocumentsMutex_);
    const auto found = materialXDocuments_.find(materialLeaf);
    if (found == materialXDocuments_.end())
        return std::nullopt;
    return found->second;
}

MaterialRef HdNoorRayRenderParam::GetNativeGreyMaterial()
{
    if (!nativeGreyMaterial_.isValid()) {
        nativeGreyMaterial_ =
            session.scene.addMaterial(GetSharedNativeFallbackMaterial());
        QueueSceneMaterialCompilation(nativeGreyMaterial_);
    }
    return nativeGreyMaterial_;
}

void HdNoorRayRenderParam::QueueSceneMaterialCompilation(const MaterialRef slot)
{
    if (slot.isValid()
        && std::ranges::find(pendingSceneMaterialCompiles_, slot)
            == pendingSceneMaterialCompiles_.end())
        pendingSceneMaterialCompiles_.push_back(slot);
}

bool HdNoorRayRenderParam::CompileSceneMaterials()
{
    std::vector<MaterialRef> toCompile;
    {
        std::scoped_lock lock(mutex);
        toCompile.swap(pendingSceneMaterialCompiles_);
    }
    if (toCompile.empty())
        return false;

    if (!session.raytracer)
        return false;
    session.rebuildNativeMaterials();
    return true;
}

void HdNoorRayRenderParam::PublishMaterial(
    const SdfPath& id, const MaterialX::DocumentPtr& document)
{
    MaterialRef& published = materials_[id];
    if (published.isValid())
    {
        session.scene.updateMaterialDocument(published.handle(), document);
    }
    else
    {
        published = session.scene.addMaterial(document);
    }
    session.scene.setDirtyFlag(Meshes);
    // Material and texture edits change the radiance represented by the
    // progressive framebuffer. Keep the next render from accumulating the
    // new shader over samples produced by the previous shader.
    session.scene.setDirtyFlag(Accumulation);

    const auto bindings = materialBindings_.find(id);
    if (bindings != materialBindings_.end())
        for (const auto& [mesh, slot] : bindings->second)
            if (MeshAsset* asset = session.scene.getMeshAsset(mesh))
                asset->setMaterial(slot, published);
}

void HdNoorRayRenderParam::PublishFallbackMaterial(
    const SdfPath& id, const MaterialX::DocumentPtr& document)
{
    std::scoped_lock lock(mutex);

    // Make every queued result for the previous custom document stale before
    // publishing the fallback, otherwise a late completion could resurrect
    // the material after its empty transport tombstone was processed.
    ++materialCompileGenerations_[id];

    PublishMaterial(id, document);

    // PublishMaterial only swaps the document in place; a stale program from
    // the previous document would otherwise keep rendering. Invalidate the
    // slot and let the render thread compile the fallback onto it -- unless
    // this exact document is already current and compiled.
    const auto published = materials_.find(id);
    if (published != materials_.end() && published->second.isValid()) {
        const MaterialHandle handle = published->second.handle();
        const auto& sceneDocuments = session.scene.getMaterialXDocuments();
        const bool alreadyCurrent = handle.index() < sceneDocuments.size()
            && sceneDocuments[handle.index()] == document
            && session.scene.getMaterial(handle).svmBytecodeLength != 0;
        if (!alreadyCurrent) {
            session.scene.invalidateMaterial(handle);
            QueueSceneMaterialCompilation(published->second);
        }
    }
}

void HdNoorRayRenderParam::ReleaseMaterial(const SdfPath& id)
{
    // Invalidate work queued by the retiring Sprim. Keep the monotonically
    // increasing counter so a new Sprim at the same path cannot accidentally
    // accept an older job with a reused generation number.
    ++materialCompileGenerations_[id];

    const auto published = materials_.find(id);
    if (published != materials_.end() && published->second.isValid()) {
        const MaterialRef& slot = published->second;

        // Meshes can outlive their material Sprim. Turn the shared slot into a
        // visible native grey fallback before dropping the delegate's
        // reference; their MaterialRefs keep it alive until Hydra either
        // rebinds or finalizes those meshes. The render thread compiles the
        // grey document onto the same slot.
        session.scene.updateMaterialDocument(slot.handle(),
            GetSharedNativeFallbackMaterial());
        session.scene.invalidateMaterial(slot.handle());
        QueueSceneMaterialCompilation(slot);
        materials_.erase(published);
    }
}

void HdNoorRayRenderParam::BindMaterial(
    const SdfPath& id, const MeshAssetHandle mesh, const uint32_t slot)
{
    if (id.IsEmpty() || !mesh.isValid())
        return;
    auto& bindings = materialBindings_[id];
    const auto entry = std::make_pair(mesh, slot);
    if (std::ranges::find(bindings, entry) == bindings.end())
        bindings.push_back(entry);

    const auto material = materials_.find(id);
    if (material == materials_.end() || !material->second.isValid())
        return;
    if (MeshAsset* asset = session.scene.getMeshAsset(mesh))
        asset->setMaterial(slot, material->second);
}

void HdNoorRayRenderParam::UnbindMaterial(
    const SdfPath& id, const MeshAssetHandle mesh, const uint32_t slot)
{
    const auto found = materialBindings_.find(id);
    if (found == materialBindings_.end())
        return;
    std::erase(found->second, std::make_pair(mesh, slot));
    if (found->second.empty())
        materialBindings_.erase(found);
}

PXR_NAMESPACE_CLOSE_SCOPE
