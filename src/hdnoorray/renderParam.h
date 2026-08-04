#pragma once

#include "api.h"

#include <pxr/imaging/hd/renderDelegate.h>
#include <pxr/usd/sdf/path.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <semaphore>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <tbb/concurrent_queue.h>
#include <tbb/task_group.h>

#include "Materials/SVM/SvmCompiler.h"

#include "NoorRaySession.h"
#include "Scene/Resources/SceneResources.h"
#include "Scene/Resources/Texture.h"
#include "Materials/Shading/Material.h"

PXR_NAMESPACE_OPEN_SCOPE

class HDNOORRAY_API HdNoorRayRenderParam final : public HdRenderParam
{
public:
    struct MaterialCompilationOutput
    {
        nr::svm::CompiledSvmProgram program;
    };

    // An immutable snapshot of MaterialX XML sent directly by the Blender
    // addon. Snapshots make lookups cheap while SetRenderSetting can replace
    // the latest document concurrently with Hydra material Sync.
    struct MaterialXDocument
    {
        std::shared_ptr<const std::string> xml;
        uint64_t hash{};
        uint64_t revision{};

        bool IsTombstone() const { return !xml || xml->empty(); }
    };

    HdNoorRayRenderParam();
    ~HdNoorRayRenderParam() noexcept override;

    HdNoorRayRenderParam(const HdNoorRayRenderParam&) = delete;
    HdNoorRayRenderParam& operator=(const HdNoorRayRenderParam&) = delete;

    void SetProgress(double progress);
    double GetProgress() const;
    void AccumulateGpuTimeMs(float ms);
    double GetTotalClockTime() const;
    void ResetClock();

    // Stores the latest in-memory document for one Hydra material leaf.
    // Empty XML is an explicit tombstone that makes Sync use Hydra's normal
    // material resource again. Returns false when the exact document was
    // already current, avoiding needless dirty-setting churn.
    bool SetMaterialXDocument(std::string materialLeaf, std::string xml);
    std::optional<MaterialXDocument> GetMaterialXDocument(
        const std::string& materialLeaf) const;

    // Textures are shared between the materials that name the same file. The
    // retained by the Scene-wide image library until the scene is cleared.
    // Returns an invalid handle when the file cannot be loaded.
    TextureHandle GetOrCreateTexture(
        const std::string& filePath, TextureEncoding encoding,
        bool flipY = true);
    // The cache stores generation-checked handles rather than owning
    // references, so it can survive CommitResources without keeping unused
    // scene textures alive. Pruning removes handles reclaimed by the Scene.
    void PruneTextureCache();
    // Hydra may Sync material Sprims concurrently. Limit the expensive
    // hashing/decode portion so large EXRs do not create an unbounded
    // temporary-memory wave.
    void AcquireMaterialSyncSlot() { materialSyncSlots_.acquire(); }
    void ReleaseMaterialSyncSlot() { materialSyncSlots_.release(); }
    void PublishMaterial(const SdfPath& id, const MaterialX::DocumentPtr& document);
    // Cancels queued compiles and replaces any compiled program immediately
    // with a native material. Used when a custom-document tombstone falls
    // back to an empty or unsupported Hydra resource, and as the last resort
    // when a MaterialX graph no longer compiles (a default MaterialX material
    // is published instead of the broken graph).
    void PublishFallbackMaterial(
        const SdfPath& id, const MaterialX::DocumentPtr& document);
    // Returns the scene's single native grey material slot, created on
    // demand. Every unbound mesh slot shares it, so the grey fallback is
    // compiled once instead of once per mesh. The document is published here
    // and the render-thread compile pass installs its program. Caller must
    // hold `mutex` (mesh Sync and PublishFallbackMaterial/ReleaseMaterial do).
    MaterialRef GetNativeGreyMaterial();
    // Marks an existing scene material slot for the render thread to compile.
    // Used by the grey fallback paths that publish a document directly into a
    // scene slot without going through QueueMaterialCompilation (which owns
    // its own Hydra material prim). Caller must hold `mutex`.
    void QueueSceneMaterialCompilation(MaterialRef slot);
    void QueueMaterialCompilation(const SdfPath& id, MaterialX::DocumentPtr document,
        std::function<MaterialCompilationOutput()> compile);
    // Returns true when a completed background compile changed renderer state.
    // This is the one kind of change that happens outside Hydra's dirty-bit
    // lifecycle, so the render pass folds it into its accumulation reset.
    bool ProcessMaterialCompilations();
    // Compiles the documents on every scene slot queued via
    // QueueSceneMaterialCompilation and stores the resulting program spans
    // back on those slots. Runs on the render thread only. Returns true when
    // any slot gained a program.
    bool CompileSceneMaterials();
    bool HasPendingMaterialCompilations() const
    {
        return pendingMaterialCompiles_.load(std::memory_order_relaxed) != 0;
    }
    void ReleaseMaterial(const SdfPath& id);
    // Binds the material a mesh prim asked for, at the given material slot
    // (0 is the mesh's own primary material; 1+ come from HdGeomSubsets, see
    // mesh.cpp's BuildFaceMaterialSlots). Meshes are addressed by handle
    // because the asset storage is a managed vector that moves its elements
    // when it grows.
    void BindMaterial(const SdfPath& id, MeshAssetHandle mesh, uint32_t slot = 0);
    void UnbindMaterial(const SdfPath& id, MeshAssetHandle mesh, uint32_t slot = 0);

    mutable std::mutex mutex;
    noorray::NoorRaySession session;

private:
    struct ContentIdentity
    {
        uint64_t first{};
        uint64_t second{};
        uint64_t size{};

        bool operator==(const ContentIdentity&) const = default;
    };

    struct ContentIdentityHash
    {
        size_t operator()(const ContentIdentity& identity) const noexcept
        {
            size_t hash = static_cast<size_t>(identity.first);
            hash ^= static_cast<size_t>(identity.second)
                + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
            hash ^= static_cast<size_t>(identity.size)
                + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
            return hash;
        }
    };

    struct TextureCacheKey
    {
        ContentIdentity content;
        TextureEncoding encoding{TextureEncoding::Linear8};
        bool flipY{true};

        bool operator==(const TextureCacheKey&) const = default;
    };

    struct TextureCacheKeyHash
    {
        size_t operator()(const TextureCacheKey& key) const noexcept
        {
            size_t hash = ContentIdentityHash{}(key.content);
            hash ^= static_cast<size_t>(key.encoding)
                + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
            hash ^= static_cast<size_t>(key.flipY)
                + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
            return hash;
        }
    };

    struct AssetFingerprint
    {
        uint64_t size{};
        uint64_t timestampBits{};
        ContentIdentity content;
    };

    enum class DecodedPixelType
    {
        Rgba8,
        Rgba16Float,
        Rgba32Float,
    };

    struct DecodedTextureData
    {
        int width{};
        int height{};
        DecodedPixelType pixelType{DecodedPixelType::Rgba8};
        std::shared_ptr<const std::vector<uint8_t>> rgba8;
        std::shared_ptr<const std::vector<uint16_t>> rgba16Float;
        std::shared_ptr<const std::vector<float>> rgba32Float;
    };

    struct DecodedTextureCacheEntry
    {
        int width{};
        int height{};
        DecodedPixelType pixelType{DecodedPixelType::Rgba8};
        std::weak_ptr<const std::vector<uint8_t>> rgba8;
        std::weak_ptr<const std::vector<uint16_t>> rgba16Float;
        std::weak_ptr<const std::vector<float>> rgba32Float;
    };

    struct DecodedTextureCacheKey
    {
        ContentIdentity content;
        bool flipY{true};

        bool operator==(const DecodedTextureCacheKey&) const = default;
    };

    struct DecodedTextureCacheKeyHash
    {
        size_t operator()(const DecodedTextureCacheKey& key) const noexcept
        {
            size_t hash = ContentIdentityHash{}(key.content);
            hash ^= static_cast<size_t>(key.flipY)
                + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
            return hash;
        }
    };

    struct MaterialCompilationResult
    {
        SdfPath id;
        uint64_t generation{};
        MaterialX::DocumentPtr document;
        MaterialCompilationOutput output;
        std::string error;
    };

    ContentIdentity GetTextureContentIdentity(const std::string& filePath);
    DecodedTextureData GetOrDecodeTexture(
        const std::string& filePath, const ContentIdentity& content,
        bool flipY);
    TextureHandle GetOrCreateMemoryTexture(
        const std::string& dataUri, TextureEncoding encoding);

    std::atomic<double> progress_{};
    std::atomic<double> cumulativeGpuTimeSeconds_{};
    mutable std::shared_mutex materialXDocumentsMutex_;
    std::unordered_map<std::string, MaterialXDocument> materialXDocuments_;
    uint64_t nextMaterialXDocumentRevision_{};
    std::map<SdfPath, MaterialRef> materials_;
    // One entry per (mesh, slot) currently bound to this material path -- a
    // mesh can appear more than once if it binds the same material at
    // several slots (or, via HdGeomSubsets, different slots point at
    // different material paths and so appear in different map entries).
    std::map<SdfPath, std::vector<std::pair<MeshAssetHandle, uint32_t>>> materialBindings_;
    std::unordered_map<std::string, AssetFingerprint>
        assetFingerprintCache_;
    std::unordered_map<std::string, std::shared_future<ContentIdentity>>
        assetHashFlights_;
    std::unordered_map<DecodedTextureCacheKey, DecodedTextureCacheEntry,
        DecodedTextureCacheKeyHash> decodedTextureCache_;
    std::unordered_map<DecodedTextureCacheKey,
        std::shared_future<DecodedTextureData>, DecodedTextureCacheKeyHash>
        decodedTextureFlights_;
    // Non-owning deduplication index. Texture bytes live only in Scene;
    // these handles are discarded when the scene invalidates them.
    std::unordered_map<TextureCacheKey, TextureHandle, TextureCacheKeyHash>
        textureCache_;
    std::counting_semaphore<64> materialSyncSlots_{4};
    std::map<SdfPath, uint64_t> materialCompileGenerations_;
    tbb::task_group materialCompileTasks_;
    tbb::concurrent_queue<MaterialCompilationResult> completedMaterialCompiles_;
    std::vector<MaterialCompilationResult> readyMaterialCompiles_;
    std::atomic<uint32_t> pendingMaterialCompiles_{};
    // The shared native grey slot handed to every mesh with no bound material.
    // Guarded by `mutex`.
    MaterialRef nativeGreyMaterial_;
    // Scene slots whose fallback documents need their program installed by the
    // render thread (ProcessMaterialCompilations). Guarded by `mutex`.
    std::vector<MaterialRef> pendingSceneMaterialCompiles_;
};

PXR_NAMESPACE_CLOSE_SCOPE
