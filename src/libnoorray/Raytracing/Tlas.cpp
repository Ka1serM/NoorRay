#include "Raytracing/Tlas.h"

#include <algorithm>
#include <cstring>

#include <optix_stubs.h>

#include "CUDA/Checks.h"
#include "Mesh/GaussianCutoff.h"
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

    const RenderSettings& rs = scene.getRenderSettings();
    const GaussianProxyType proxyType = rs.gaussianProxyType;
    const float cutoffSigma = rs.gaussianCutoffSigma;
    if (proxyBlas == nullptr || lastProxyType != proxyType || lastCutoffSigma != cutoffSigma)
    {
        if (proxyBlas == nullptr)
            proxyBlas = std::make_unique<GaussianProxyBlas>();
        proxyBlas->build(context, stream, proxyType, cutoffSigma);
        lastProxyType = proxyType;
        lastCutoffSigma = cutoffSigma;
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
            // Bake the complete scene-instance TRS and the Gaussian asset's
            // translation/rotation/sigma scale into one OptiX instance matrix.
            // The shared proxy BLAS carries the maximum cutoff scaling. Keep
            // the TLAS transform in true Gaussian Mahalanobis space.
            const mat4 gaussToInstance = toMat4(gaussians[i].transform);
            const mat4 worldTransform = instanceToWorld * gaussToInstance;
            inputs.push_back({toOptixTransform(worldTransform), proxyBlas->getTraversable(), gaussianGlobalId});
            ++gaussianGlobalId;
        }
    }
}

void Tlas::updateMeshInstanceInPlace(
    const Scene& scene,
    nr::rstd::vector<GpuInstance>& instancesOut,
    const uint32_t index)
{
    const auto meshInstances = scene.getMeshInstances();
    const MeshInstance& instance = *meshInstances[index];
    const mat4 objectToWorld = instance.getWorldTransform().getMatrix();
    const mat4 worldToObject = inverse(objectToWorld);
    const uint32_t meshIndex = instance.getMeshIndex();
    const auto& asset = scene.getMeshAsset(meshIndex);

    GpuInstance gpuInstance{};
    gpuInstance.objectToWorld = objectToWorld;
    gpuInstance.normalToWorld = glm::mat3(transpose(worldToObject));
    gpuInstance.meshIndex = meshIndex;
    instancesOut[index] = gpuInstance;

    // instanceBuffer is managed memory: writing here is immediately visible
    // to the GPU, no upload needed. Safe because the caller (Raytracer::
    // updateTLAS) synchronizes the stream before touching this buffer.
    OptixInstance& destination = instancePtr()[index];
    const auto transform = toOptixTransform(objectToWorld);
    std::copy(transform.begin(), transform.end(), destination.transform);
    destination.instanceId = index;
    destination.sbtOffset = 0;
    destination.visibilityMask = 0x01;
    destination.flags = OPTIX_INSTANCE_FLAG_NONE;
    destination.traversableHandle = asset.getBlas().getTraversable();
}

bool Tlas::tryPartialUpdate(
    const OptixDeviceContext context,
    const cudaStream_t stream,
    const Scene& scene,
    nr::rstd::vector<GpuInstance>& instancesOut)
{
    const auto& dirtyIndices = scene.getDirtyMeshInstanceIndices();
    if (tlasHandle == 0 || !instanceBuffer || dirtyIndices.empty())
        return false;

    const auto meshInstances = scene.getMeshInstances();
    if (meshInstances.size() != meshInstanceCount
        || static_cast<uint32_t>(instancesOut.size()) != meshInstanceCount
        || scene.getGaussianCount() != instanceCount - meshInstanceCount)
        return false;

    for (const uint32_t index : dirtyIndices)
        if (index >= meshInstanceCount)
            return false;

    for (const uint32_t index : dirtyIndices)
        updateMeshInstanceInPlace(scene, instancesOut, index);

    OptixBuildInput buildInput{};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    buildInput.instanceArray.instances = instanceBuffer.devicePtr();
    buildInput.instanceArray.numInstances = instanceCount;

    OptixAccelBuildOptions options{};
    options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    options.operation = OPTIX_BUILD_OPERATION_UPDATE;

    OptixAccelBufferSizes sizes{};
    NR_OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &options, &buildInput, 1, &sizes));

    nr::cuda::UniqueAsyncDeviceBuffer scratch(sizes.tempUpdateSizeInBytes, stream);
    NR_OPTIX_CHECK(optixAccelBuild(
        context,
        stream,
        &options,
        &buildInput,
        1,
        scratch.devicePtr(),
        sizes.tempUpdateSizeInBytes,
        tlasBuffer.devicePtr(),
        tlasBufferSize,
        &tlasHandle,
        nullptr,
        0));
    return true;
}

void Tlas::build(
    const OptixDeviceContext context,
    const cudaStream_t stream,
    const Scene& scene,
    nr::rstd::vector<GpuInstance>& instancesOut)
{
    if (tryPartialUpdate(context, stream, scene, instancesOut))
        return;

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

    // instanceBuffer is managed (UMA) memory so the instance array can be
    // updated in place by tryPartialUpdate() without any host->device copy.
    const size_t instanceBytes = optixInstances.size() * sizeof(OptixInstance);
    if (!instanceBuffer || instanceCount != instances.size())
    {
        instanceBuffer.allocate(instanceBytes);
    }
    std::memcpy(reinterpret_cast<void*>(instanceBuffer.devicePtr()), optixInstances.data(), instanceBytes);

    OptixBuildInput buildInput{};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    buildInput.instanceArray.instances = instanceBuffer.devicePtr();
    buildInput.instanceArray.numInstances = static_cast<unsigned int>(instances.size());
    OptixAccelBuildOptions options{};
    options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    options.operation = operation;
    OptixAccelBufferSizes sizes{};
    NR_OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &options, &buildInput, 1, &sizes));

    if (operation == OPTIX_BUILD_OPERATION_BUILD && sizes.outputSizeInBytes > tlasBufferSize)
    {
        tlasBuffer.allocate(sizes.outputSizeInBytes, stream);
        tlasBufferSize = sizes.outputSizeInBytes;
    }

    const size_t scratchSize = operation == OPTIX_BUILD_OPERATION_UPDATE
        ? sizes.tempUpdateSizeInBytes
        : sizes.tempSizeInBytes;
    nr::cuda::UniqueAsyncDeviceBuffer scratch(scratchSize, stream);
    NR_OPTIX_CHECK(optixAccelBuild(
        context,
        stream,
        &options,
        &buildInput,
        1,
        scratch.devicePtr(),
        scratchSize,
        tlasBuffer.devicePtr(),
        tlasBufferSize,
        &tlasHandle,
        nullptr,
        0));
    instanceCount = static_cast<uint32_t>(instances.size());
}

void Tlas::reset() noexcept
{
    proxyBlas.reset();
    instanceBuffer.reset();
    tlasBuffer.reset();
    tlasBufferSize = 0;
    instanceCount = 0;
    tlasHandle = 0;
}
