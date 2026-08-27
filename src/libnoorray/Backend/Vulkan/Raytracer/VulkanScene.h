#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <gpu/gpu.hpp>

class Scene;

// Packed scene-record ABI consumed by Raytracer.slang. Offsets are byte offsets
// into one descriptor-heap storage buffer; keeping the records scalar avoids
// matrix-layout and glm alignment differences between C++ and Slang.
struct VulkanSceneHeader
{
    uint32_t meshCount{};
    uint32_t instanceCount{};
    uint32_t meshOffset{};
    uint32_t instanceOffset{};
    uint32_t vertexDataOffset{};
    uint32_t indexDataOffset{};
    uint32_t faceDataOffset{};
    uint32_t totalBytes{};
};

struct VulkanSceneMeshRecord
{
    uint32_t vertexOffset{};
    uint32_t indexOffset{};
    uint32_t faceOffset{};
    uint32_t vertexCount{};
    uint32_t indexCount{};
    uint32_t faceCount{};
    // Material ids are scene-global registry slots. Faces contain a local
    // material slot, so the shader first indexes this table and then the
    // packed material-record buffer.
    uint32_t materialOffset{};
    uint32_t materialCount{};
};

struct VulkanSceneInstanceRecord
{
    float objectToWorld[12]{};
    uint32_t meshIndex{};
    uint32_t reserved[3]{};
};

static_assert(sizeof(VulkanSceneHeader) == 32);
static_assert(sizeof(VulkanSceneMeshRecord) == 32);
static_assert(sizeof(VulkanSceneInstanceRecord) == 64);

// Native scene acceleration state. Mesh geometry is copied from MeshAsset's
// backend-neutral host mirror, then each unique asset gets a Vulkan BLAS and
// each MeshInstance becomes a TLAS instance. Rebuilding is transactional from
// the renderer's point of view: callers replace the immutable object only
// after all builds and device waits have completed.
class VulkanScene
{
public:
    VulkanScene(gpu::Device& gpu_device, const Scene& scene);
    ~VulkanScene() = default;

    VulkanScene(const VulkanScene&) = delete;
    VulkanScene& operator=(const VulkanScene&) = delete;
    VulkanScene(VulkanScene&&) noexcept = default;
    VulkanScene& operator=(VulkanScene&&) noexcept = default;

    gpu::AccelerationStructureHandle topLevelHandle() const { return tlas_.handle(); }
    const gpu::Buffer<std::byte>& sceneData() const { return sceneData_; }
    uint32_t meshCount() const { return static_cast<uint32_t>(meshes.size()); }
    uint32_t instanceCount() const { return instanceCount_; }
    const gpu::Buffer<std::byte>& gaussianRecords() const { return gaussianRecords_; }
    const gpu::Buffer<std::byte>& gaussianOpacities() const { return gaussianOpacities_; }
    const gpu::Buffer<std::byte>& gaussianShCoefficients() const { return gaussianShCoefficients_; }
    const gpu::Buffer<std::byte>& gaussianInstanceOffsets() const { return gaussianInstanceOffsets_; }
    uint32_t gaussianCount() const { return gaussianCount_; }
    uint32_t gaussianShCoefficientCount() const { return gaussianShCoefficientCount_; }
    uint32_t gaussianProxyTriangleCount() const { return gaussianProxyTriangleCount_; }
    // Gaussian instances are appended after the mesh instances, so this is
    // the TLAS index the first Gaussian occupies.
    uint32_t meshInstanceCount() const { return meshInstanceCount_; }
    // Updates fixed-topology editor mutations without replacing descriptors,
    // geometry BLASes, or the TLAS allocation. Returns false when topology or
    // proxy configuration changed and a transactional rebuild is required.
    bool updateMutableData(const Scene& scene, bool updateGaussians = true);

private:
    struct MeshState
    {
        gpu::Buffer<gpu::float3> positions;
        gpu::Buffer<std::uint32_t> indices;
        gpu::AccelerationStructure blas;
    };

    struct GaussianState
    {
        gpu::Buffer<gpu::float3> positions;
        gpu::Buffer<std::uint32_t> indices;
        gpu::AccelerationStructure blas;
    };

    gpu::Device& gpuDevice;
    std::vector<MeshState> meshes;
    GaussianState gaussianProxy;
    std::vector<gpu::float4x4> gaussianTransforms;
    std::vector<gpu::Instance> tlasInstances;
    std::vector<uint32_t> meshInstanceAssetIndices;
    gpu::Buffer<std::byte> sceneData_;
    gpu::AccelerationStructure tlas_;
    uint32_t instanceCount_{};
    gpu::Buffer<std::byte> gaussianRecords_;
    gpu::Buffer<std::byte> gaussianOpacities_;
    gpu::Buffer<std::byte> gaussianShCoefficients_;
    gpu::Buffer<std::byte> gaussianInstanceOffsets_;
    uint32_t gaussianCount_{};
    uint32_t gaussianShCoefficientCount_{};
    uint32_t gaussianProxyTriangleCount_{8u};
    uint32_t meshInstanceCount_{};
    uint32_t gaussianInstanceCount_{};
    uint32_t sceneInstanceOffset_{};
    uint32_t gaussianProxyType_{};
    float gaussianCutoffSigma_{};

    void build(const Scene& scene);
    uint32_t buildMesh(const class MeshAsset& asset);
    void buildGaussians(const Scene& scene);
    void buildTopLevel(const Scene& scene);
    void buildSceneData(const Scene& scene);
};
