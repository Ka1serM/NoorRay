#include "Raytracing/Acceleration/Tlas.h"

#include <algorithm>

#include <optix_stubs.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include "CUDA/Checks.h"
#include "Mesh/Assets/MeshAsset.h"
#include "Mesh/Assets/GaussianAsset.h"
#include "Scene/GaussianInstance.h"
#include "Scene/MeshInstance.h"
#include "Scene/Scene.h"

namespace
{
std::array<float, 12> toOptixTransform(const glm::mat4& matrix)
{
    // OptiX expects a row-major 3x4 affine matrix. GLM's matrix indexing is
    // column-major by convention, so explicitly serialize rows here rather
    // than relying on glm::value_ptr().
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
    result.resize(instances.size());
    instancesOut.resize(instances.size());
    tbb::parallel_for(tbb::blocked_range<size_t>(0, instances.size(), 256),
        [&](const tbb::blocked_range<size_t>& range)
    {
        for (size_t index = range.begin(); index != range.end(); ++index)
        {
            const MeshInstance& instance = *instances[index];
            const mat4 objectToWorld = instance.getWorldTransform().getMatrix();
            const mat4 worldToObject = inverse(objectToWorld);
            const uint32_t meshIndex = instance.getMeshIndex();
            const MeshAsset& asset = instance.getMeshAsset();
            result[index] = AccelInstanceInput{toOptixTransform(objectToWorld),
                asset.getBlas().getTraversable(), static_cast<uint32_t>(index), 0,
                MeshVisibility};

            GpuInstance gpuInstance{};
            gpuInstance.objectToWorld = objectToWorld;
            gpuInstance.normalToWorld = glm::mat3(transpose(worldToObject));
            gpuInstance.meshIndex = meshIndex;
            instancesOut[index] = gpuInstance;
        }
    });
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
    {
        // Removing the last splat instance has to give the device memory back.
        // The per-instance IAS holds one OptixInstance per Gaussian plus the
        // built acceleration structure, which is the bulk of a splat scene's
        // VRAM footprint, and the proxy BLAS is useless without them.
        gaussianInstanceAccels.clear();
        gaussianInstanceCount = 0;
        proxyBlas.reset();
        return;
    }
    const RenderSettings& rs = scene.getRenderSettings();
    const GaussianProxyType proxyType = rs.gaussianProxyType;
    const float cutoffSigma = rs.gaussianCutoffSigma;
    const bool proxyChanged = proxyBlas == nullptr
        || lastProxyType != proxyType || lastCutoffSigma != cutoffSigma;
    if (proxyChanged)
    {
        if (proxyBlas == nullptr)
            proxyBlas = std::make_unique<GaussianProxyBlas>();
        proxyBlas->build(context, stream, proxyType, cutoffSigma);
        lastProxyType = proxyType;
        lastCutoffSigma = cutoffSigma;
    }

    const auto instances = scene.getGaussianInstances();
    if (gaussianInstanceAccels.size() != instances.size())
        gaussianInstanceAccels.resize(instances.size());

    uint32_t globalOffset = 0;
    for (size_t index = 0; index < instances.size(); ++index)
    {
        const uint32_t count = instances[index]->getGaussianAsset().getGaussianCount();
        const GaussianAsset& asset = instances[index]->getGaussianAsset();
        GaussianInstanceAccel& accel = gaussianInstanceAccels[index];
        if (proxyChanged || accel.objectHandle != instances[index]->getHandle()
            || accel.gaussianCount != count || accel.globalOffset != globalOffset
            || asset.isDirty())
        {
            buildGaussianInstanceAccel(
                context, stream, accel, *instances[index], globalOffset);
        }
        globalOffset += count;
    }

    const uint32_t instanceStart = static_cast<uint32_t>(inputs.size());
    inputs.resize(instanceStart + instances.size());
    tbb::parallel_for(tbb::blocked_range<size_t>(0, instances.size(), 64),
        [&](const tbb::blocked_range<size_t>& range)
    {
        for (size_t index = range.begin(); index != range.end(); ++index)
        {
            inputs[instanceStart + index] = {
                toOptixTransform(instances[index]->getWorldTransform().getMatrix()),
                gaussianInstanceAccels[index].handle,
                static_cast<uint32_t>(index), 0, GaussianVisibility};
        }
    });
    gaussianInstanceCount = static_cast<uint32_t>(instances.size());
}

void Tlas::buildGaussianInstanceAccel(
    const OptixDeviceContext context,
    const cudaStream_t stream,
    GaussianInstanceAccel& destination,
    const GaussianInstance& instance,
    const uint32_t globalOffset)
{
    const GaussianAsset& asset = instance.getGaussianAsset();
    const uint32_t gaussianCount = asset.getGaussianCount();
    const auto& gaussians = asset.getGaussians();
    const size_t instanceBytes = static_cast<size_t>(gaussianCount) * sizeof(OptixInstance);
    destination.instanceBuffer.allocate(instanceBytes);
    auto* optixInstances = reinterpret_cast<OptixInstance*>(
        destination.instanceBuffer.devicePtr());
    tbb::parallel_for(tbb::blocked_range<size_t>(0, gaussianCount, 1024),
        [&](const tbb::blocked_range<size_t>& range)
    {
        for (size_t index = range.begin(); index != range.end(); ++index)
        {
            OptixInstance& child = optixInstances[index];
            child = {};
            const auto transform = toOptixTransform(toMat4(gaussians[index].transform));
            std::copy(transform.begin(), transform.end(), child.transform);
            child.instanceId = 0;
            child.sbtOffset = 1;
            child.visibilityMask = GaussianVisibility;
            child.flags = OPTIX_INSTANCE_FLAG_NONE;
            child.traversableHandle = proxyBlas->getTraversable();
        }
    });

    OptixBuildInput buildInput{};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    buildInput.instanceArray.instances = destination.instanceBuffer.devicePtr();
    buildInput.instanceArray.numInstances = gaussianCount;
    OptixAccelBuildOptions options{};
    options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    options.operation = OPTIX_BUILD_OPERATION_BUILD;
    OptixAccelBufferSizes sizes{};
    NR_OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &options, &buildInput, 1, &sizes));
    destination.accelBuffer.allocate(sizes.outputSizeInBytes, stream);
    nr::cuda::UniqueAsyncDeviceBuffer scratch(sizes.tempSizeInBytes, stream);
    NR_OPTIX_CHECK(optixAccelBuild(
        context, stream, &options, &buildInput, 1,
        scratch.devicePtr(), sizes.tempSizeInBytes,
        destination.accelBuffer.devicePtr(), sizes.outputSizeInBytes,
        &destination.handle, nullptr, 0));
    NR_GPU_CHECK(cudaStreamSynchronize(stream));
    destination.instanceBuffer.reset();
    destination.objectHandle = instance.getHandle();
    destination.gaussianCount = gaussianCount;
    destination.globalOffset = globalOffset;
}

void Tlas::updateMeshInstanceInPlace(
    const Scene& scene,
    const std::vector<std::shared_ptr<MeshInstance>>& meshInstances,
    nr::rstd::vector<GpuInstance>& instancesOut,
    const uint32_t index)
{
    const MeshInstance& instance = *meshInstances[index];
    const mat4 objectToWorld = instance.getWorldTransform().getMatrix();
    const mat4 worldToObject = inverse(objectToWorld);
    const uint32_t meshIndex = instance.getMeshIndex();

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
    destination.visibilityMask = MeshVisibility;
    destination.flags = OPTIX_INSTANCE_FLAG_NONE;
    destination.traversableHandle =
        instance.getMeshAsset().getBlas().getTraversable();
}

bool Tlas::tryPartialUpdate(
    const OptixDeviceContext context,
    const cudaStream_t stream,
    const Scene& scene,
    nr::rstd::vector<GpuInstance>& instancesOut)
{
    const auto& dirtyIndices = scene.getDirtyMeshInstanceIndices();
    const auto& dirtyGaussianIndices = scene.getDirtyGaussianInstanceIndices();
    if (tlasHandle == 0 || !instanceBuffer
        || (dirtyIndices.empty() && dirtyGaussianIndices.empty()))
        return false;
    if (!dirtyGaussianIndices.empty())
        return false;

    const auto meshInstances = scene.getMeshInstances();
    const auto gaussianInstances = scene.getGaussianInstances();
    if (meshInstances.size() != meshInstanceCount
        || static_cast<uint32_t>(instancesOut.size()) != meshInstanceCount
        || gaussianInstances.size() != gaussianInstanceCount)
        return false;

    for (const uint32_t index : dirtyIndices)
        if (index >= meshInstanceCount)
            return false;

    tbb::parallel_for(tbb::blocked_range<size_t>(0, dirtyIndices.size(), 64),
        [&](const tbb::blocked_range<size_t>& range)
    {
        for (size_t dirty = range.begin(); dirty != range.end(); ++dirty)
            updateMeshInstanceInPlace(
                scene, meshInstances, instancesOut, dirtyIndices[dirty]);
    });
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

    // instanceBuffer is managed (UMA) memory so the instance array can be
    // updated in place by tryPartialUpdate() without any host->device copy.
    const size_t instanceBytes = instances.size() * sizeof(OptixInstance);
    if (!instanceBuffer || instanceCount != instances.size())
        instanceBuffer.allocate(instanceBytes);

    OptixInstance* optixInstances = instancePtr();
    tbb::parallel_for(tbb::blocked_range<size_t>(0, instances.size(), 1024),
        [&](const tbb::blocked_range<size_t>& range)
    {
        for (size_t index = range.begin(); index != range.end(); ++index)
        {
            const AccelInstanceInput& source = instances[index];
            OptixInstance& destination = optixInstances[index];
            destination = {};
            std::copy(source.transform.begin(), source.transform.end(), destination.transform);
            destination.instanceId = source.instanceId;
            destination.sbtOffset = source.sbtOffset;
            destination.visibilityMask = source.visibilityMask;
            destination.flags = OPTIX_INSTANCE_FLAG_NONE;
            destination.traversableHandle = source.blasHandle;
        }
    });

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
    gaussianInstanceAccels.clear();
    proxyBlas.reset();
    instanceBuffer.reset();
    tlasBuffer.reset();
    tlasBufferSize = 0;
    instanceCount = 0;
    meshInstanceCount = 0;
    gaussianInstanceCount = 0;
    tlasHandle = 0;
}
