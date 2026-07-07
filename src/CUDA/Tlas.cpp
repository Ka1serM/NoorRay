#include "CUDA/Tlas.h"

#include <algorithm>
#include <cstring>

#include <optix_stubs.h>

#include "CUDA/Checks.h"
#include "CUDA/GaussianProxyBlas.h"
#include "Mesh/MeshAsset.h"
#include "Mesh/GaussianAsset.h"
#include "Scene/GaussianInstance.h"
#include "Scene/MeshInstance.h"
#include "Scene/Scene.h"

namespace
{
std::array<float, 12> toOptixTransform(const glm::mat4& matrix)
{
    return {matrix[0][0], matrix[1][0], matrix[2][0], matrix[3][0],
            matrix[0][1], matrix[1][1], matrix[2][1], matrix[3][1],
            matrix[0][2], matrix[1][2], matrix[2][2], matrix[3][2]};
}

glm::mat4 toMat4(const glm::mat4x3& m)
{
    glm::mat4 result(1.0f);
    result[0] = glm::vec4(m[0], 0.0f);
    result[1] = glm::vec4(m[1], 0.0f);
    result[2] = glm::vec4(m[2], 0.0f);
    result[3] = glm::vec4(m[3], 1.0f);
    return result;
}
}

std::vector<AccelInstanceInput> Tlas::buildInstanceInputs(
    const Scene& scene,
    nr::rstd::vector<GpuInstance>& instancesOut) const
{
    std::vector<AccelInstanceInput> result;
    const auto instances = scene.getMeshInstances();
    result.reserve(instances.size());
    instancesOut.clear();
    instancesOut.reserve(instances.size());
    for (uint32_t index = 0; index < instances.size(); ++index)
    {
        const MeshInstance& instance = *instances[index];
        const mat4 objectToWorld = instance.getWorldTransform().getMatrix();
        const mat4 worldToObject = inverse(objectToWorld);
        const uint32_t meshIndex = instance.getMeshIndex();
        const auto& asset = scene.getMeshAsset(meshIndex);
        result.push_back({
            toOptixTransform(objectToWorld), asset.getBlas().getTraversable(), index});

        GpuInstance gpuInstance{};
        gpuInstance.objectToWorld = objectToWorld;
        gpuInstance.normalToWorld = glm::mat3(transpose(worldToObject));
        gpuInstance.meshIndex = meshIndex;
        instancesOut.push_back(gpuInstance);
    }
    return result;
}

void Tlas::buildGaussianInstances(
    const OptixDeviceContext context,
    const cudaStream_t stream,
    const Scene& scene,
    std::vector<AccelInstanceInput>& inputs)
{
    const uint32_t gaussianCount = scene.getGaussianCount();
    if (gaussianCount == 0)
        return;

    const GaussianProxyType proxyType =
        static_cast<GaussianProxyType>(scene.getRenderSettings().gaussianProxyType);

    if (proxyBlas == nullptr || lastProxyType != proxyType)
    {
        if (proxyBlas == nullptr)
            proxyBlas = new GaussianProxyBlas();
        proxyBlas->build(context, stream, proxyType);
        lastProxyType = proxyType;
    }

    const uint32_t instanceStart = static_cast<uint32_t>(inputs.size());
    inputs.reserve(instanceStart + gaussianCount);

    const auto instances = scene.getGaussianInstances();
    uint32_t gaussianGlobalId = 0;
    for (const auto& instance : instances)
    {
        const mat4 instanceToWorld = instance->getWorldTransform().getMatrix();
        const GaussianAsset& asset = instance->getGaussianAsset();
        const auto& gaussians = asset.getGaussians();
        for (uint32_t i = 0; i < asset.getGaussianCount(); ++i)
        {
            const mat4 gaussToInstance = toMat4(gaussians[i].transform);
            const mat4 worldTransform = instanceToWorld * gaussToInstance;
            inputs.push_back({toOptixTransform(worldTransform), proxyBlas->getTraversable(), gaussianGlobalId});
            ++gaussianGlobalId;
        }
    }
}

void Tlas::build(
    const OptixDeviceContext context,
    const cudaStream_t stream,
    const Scene& scene,
    nr::rstd::vector<GpuInstance>& instancesOut)
{
    auto accelInstances = buildInstanceInputs(scene, instancesOut);
    meshInstanceCount = static_cast<uint32_t>(instancesOut.size());

    buildGaussianInstances(context, stream, scene, accelInstances);

    const OptixBuildOperation operation = accelInstances.size() != instanceCount
        ? OPTIX_BUILD_OPERATION_BUILD
        : OPTIX_BUILD_OPERATION_UPDATE;

    buildInternal(context, stream, accelInstances, operation);
}

void Tlas::buildInternal(
    const OptixDeviceContext context,
    const cudaStream_t stream,
    const std::vector<AccelInstanceInput>& instances,
    const OptixBuildOperation operation)
{
    if (instances.empty())
    {
        tlasHandle = 0;
        return;
    }

    std::vector<OptixInstance> optixInstances(instances.size());
    for (size_t index = 0; index < instances.size(); ++index)
    {
        const AccelInstanceInput& source = instances[index];
        OptixInstance& destination = optixInstances[index];
        std::copy(source.transform.begin(), source.transform.end(), destination.transform);
        destination.instanceId = source.instanceId;
        destination.sbtOffset = index < meshInstanceCount ? 0 : 1;
        destination.visibilityMask = index < meshInstanceCount ? 0x01 : 0x02;
        // Backface culling forced on for every instance (mesh and Gaussian
        // proxy alike) via the ray-flag-driven culling in RayTraversal.h.
        destination.flags = OPTIX_INSTANCE_FLAG_NONE;
        destination.traversableHandle = source.blasHandle;
    }

    const size_t instanceBytes = optixInstances.size() * sizeof(OptixInstance);
    if (instanceBuffer == 0 || instanceCount != instances.size())
    {
        if (instanceBuffer != 0)
            NR_GPU_CHECK(cudaFreeAsync(reinterpret_cast<void*>(instanceBuffer), stream));
        void* allocation = nullptr;
        NR_GPU_CHECK(cudaMallocAsync(&allocation, instanceBytes, stream));
        instanceBuffer = reinterpret_cast<CUdeviceptr>(allocation);
    }
    NR_GPU_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(instanceBuffer),
        optixInstances.data(),
        instanceBytes,
        cudaMemcpyHostToDevice,
        stream));

    OptixBuildInput buildInput{};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    buildInput.instanceArray.instances = instanceBuffer;
    buildInput.instanceArray.numInstances = static_cast<unsigned int>(instances.size());
    OptixAccelBuildOptions options{};
    options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    options.operation = operation;
    OptixAccelBufferSizes sizes{};
    NR_OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &options, &buildInput, 1, &sizes));

    if (operation == OPTIX_BUILD_OPERATION_BUILD && sizes.outputSizeInBytes > tlasBufferSize)
    {
        if (tlasBuffer != 0)
            NR_GPU_CHECK(cudaFreeAsync(reinterpret_cast<void*>(tlasBuffer), stream));
        void* allocation = nullptr;
        NR_GPU_CHECK(cudaMallocAsync(&allocation, sizes.outputSizeInBytes, stream));
        tlasBuffer = reinterpret_cast<CUdeviceptr>(allocation);
        tlasBufferSize = sizes.outputSizeInBytes;
    }

    const size_t scratchSize = operation == OPTIX_BUILD_OPERATION_UPDATE
        ? sizes.tempUpdateSizeInBytes
        : sizes.tempSizeInBytes;
    void* scratch = nullptr;
    NR_GPU_CHECK(cudaMallocAsync(&scratch, scratchSize, stream));
    NR_OPTIX_CHECK(optixAccelBuild(
        context,
        stream,
        &options,
        &buildInput,
        1,
        reinterpret_cast<CUdeviceptr>(scratch),
        scratchSize,
        tlasBuffer,
        tlasBufferSize,
        &tlasHandle,
        nullptr,
        0));
    NR_GPU_CHECK(cudaFreeAsync(scratch, stream));
    instanceCount = static_cast<uint32_t>(instances.size());
}

void Tlas::destroy(const cudaStream_t stream) noexcept
{
    if (proxyBlas != nullptr)
    {
        proxyBlas->destroy(stream);
        delete proxyBlas;
        proxyBlas = nullptr;
    }
    if (instanceBuffer != 0)
        cudaFreeAsync(reinterpret_cast<void*>(instanceBuffer), stream);
    if (tlasBuffer != 0)
        cudaFreeAsync(reinterpret_cast<void*>(tlasBuffer), stream);
    instanceBuffer = 0;
    tlasBuffer = 0;
    tlasBufferSize = 0;
    instanceCount = 0;
    tlasHandle = 0;
}
