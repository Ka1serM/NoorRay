#include "WavefrontRaytracer.h"

#include "Camera/CameraBase.h"
#include "Mesh/MeshAsset.h"
#include "RayLUT.h"
#include "Scene/MeshInstance.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr uint32_t WavefrontGroupSize = 128u;
}

WavefrontRaytracer::WavefrontRaytracer(Scene& scene, uint32_t width, uint32_t height)
    : Raytracer(scene, width, height)
{
    rayLutGenerator = std::make_unique<RayLutGenerator>(context);
    updateDispatchCounts();
}

WavefrontRaytracer::~WavefrontRaytracer()
{
    context.getDevice().waitIdle();
}

// ── Descriptor set 0 (scene resources) ──────────────────────────────────────

void WavefrontRaytracer::createSceneDescriptorSet(
    const std::vector<vk::DescriptorSetLayoutBinding>& bindings)
{
    const uint32_t lastIdx = static_cast<uint32_t>(bindings.size() - 1);
    std::vector<vk::DescriptorBindingFlags> flags(bindings.size());
    flags[lastIdx] =
        vk::DescriptorBindingFlagBits::ePartiallyBound |
        vk::DescriptorBindingFlagBits::eVariableDescriptorCount |
        vk::DescriptorBindingFlagBits::eUpdateAfterBind;

    vk::DescriptorSetLayoutBindingFlagsCreateInfo flagsInfo{};
    flagsInfo.setBindingFlags(flags);

    vk::DescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.setBindings(bindings);
    layoutInfo.setPNext(&flagsInfo);
    layoutInfo.setFlags(vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool);
    descSetLayout = context.getDevice().createDescriptorSetLayoutUnique(layoutInfo);

    vk::DescriptorSetVariableDescriptorCountAllocateInfo varCount{};
    varCount.descriptorSetCount = 1;
    varCount.pDescriptorCounts  = &MAX_TEXTURES;

    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo.setDescriptorPool(context.getDescriptorPool());
    allocInfo.setSetLayouts(descSetLayout.get());
    allocInfo.setPNext(&varCount);
    descriptorSet = std::move(context.getDevice().allocateDescriptorSetsUnique(allocInfo).front());
}

// ── Descriptor set 1 (wavefront queues) ─────────────────────────────────────

void WavefrontRaytracer::createWavefrontSet()
{
    // Bindings 0-5: counters, pathStates, inputRayQueue, outputRayQueue, hitQueue, shadowQueue
    const std::vector<vk::DescriptorSetLayoutBinding> set1Bindings {
        {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {4, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {5, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
    };

    vk::DescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.setBindings(set1Bindings);
    wavefrontSetLayout = context.getDevice().createDescriptorSetLayoutUnique(layoutInfo);

    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo.setDescriptorPool(context.getDescriptorPool());
    allocInfo.setSetLayouts(wavefrontSetLayout.get());
    wavefrontDescSet = std::move(context.getDevice().allocateDescriptorSetsUnique(allocInfo).front());

    const vk::DeviceSize n = maxPixels();
    countersBuffer   = Buffer(context, Buffer::Type::Storage, 4 * sizeof(int32_t));
    pathStatesBuffer = Buffer(context, Buffer::Type::Storage, n * 112); // 7 × (float3+uint) Std430
    rayQueueBuffer   = Buffer(context, Buffer::Type::Storage, n * 32);  // PathRayWorkItem, shared input/output
    hitQueueBuffer   = Buffer(context, Buffer::Type::Storage, n * 48);  // HitWorkItem Std430 stride
    shadowQueueBuffer= Buffer(context, Buffer::Type::Storage, n * 2 * 48); // ShadowWorkItem × 2
}

void WavefrontRaytracer::writeWavefrontDescriptors()
{
    // rayQueueBuffer is bound to both binding 2 (inputRayQueue) and binding 3 (outputRayQueue).
    // This is safe because Extend fully drains the queue before Shade writes continuation rays.
    vk::DescriptorBufferInfo rayQueueInfo = rayQueueBuffer.getDescriptorInfo();
    std::array<vk::DescriptorBufferInfo, 6> infos {
        countersBuffer.getDescriptorInfo(),
        pathStatesBuffer.getDescriptorInfo(),
        rayQueueInfo,  // binding 2: inputRayQueue
        rayQueueInfo,  // binding 3: outputRayQueue (same buffer)
        hitQueueBuffer.getDescriptorInfo(),
        shadowQueueBuffer.getDescriptorInfo(),
    };
    std::array<vk::WriteDescriptorSet, 6> writes;
    for (uint32_t i = 0; i < 6; ++i) {
        writes[i] = vk::WriteDescriptorSet{}
            .setDstSet(wavefrontDescSet.get())
            .setDstBinding(i)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setDescriptorCount(1)
            .setBufferInfo(infos[i]);
    }
    context.getDevice().updateDescriptorSets(writes, {});
}

// ── Pipeline layout + pipelines ─────────────────────────────────────────────

void WavefrontRaytracer::buildPipelineLayout()
{
    std::array<vk::DescriptorSetLayout, 2> setLayouts {
        descSetLayout.get(), wavefrontSetLayout.get()
    };

    vk::PushConstantRange pcRange{};
    pcRange.setStageFlags(vk::ShaderStageFlagBits::eCompute);
    pcRange.setSize(sizeof(PushData));

    vk::PipelineLayoutCreateInfo info{};
    info.setSetLayouts(setLayouts);
    info.setPushConstantRanges(pcRange);
    pipelineLayout = context.getDevice().createPipelineLayoutUnique(info);
}

void WavefrontRaytracer::buildPipelines(const std::array<SpvEntry, 6>& spvs)
{
    for (uint32_t i = 0; i < 6; ++i) {
        auto mod = context.getDevice().createShaderModuleUnique(
            {{}, spvs[i].size, reinterpret_cast<const uint32_t*>(spvs[i].data)});

        vk::PipelineShaderStageCreateInfo stageInfo{};
        stageInfo.setStage(vk::ShaderStageFlagBits::eCompute);
        stageInfo.setModule(*mod);
        stageInfo.setPName("main");

        vk::ComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.setStage(stageInfo);
        pipelineInfo.setLayout(pipelineLayout.get());

        auto result = context.getDevice().createComputePipelineUnique({}, pipelineInfo);
        if (result.result != vk::Result::eSuccess)
            throw std::runtime_error("Failed to create wavefront pipeline " + std::to_string(i));
        pipelines[i] = std::move(result.value);
    }
}

// ── bindOutputImages ─────────────────────────────────────────────────────────

void WavefrontRaytracer::bindOutputImages()
{
    // Set 0 bindings 1-6: outputColor, outputAlbedo, outputNormal, outputCrypto, outputPosition, outputMaterial
    std::array<vk::DescriptorImageInfo, 6> imageInfos {
        vk::DescriptorImageInfo({}, outputColor.getView(),    vk::ImageLayout::eGeneral),
        vk::DescriptorImageInfo({}, outputAlbedo.getView(),   vk::ImageLayout::eGeneral),
        vk::DescriptorImageInfo({}, outputNormal.getView(),   vk::ImageLayout::eGeneral),
        vk::DescriptorImageInfo({}, outputCrypto.getView(),   vk::ImageLayout::eGeneral),
        vk::DescriptorImageInfo({}, outputPosition.getView(), vk::ImageLayout::eGeneral),
        vk::DescriptorImageInfo({}, outputMaterial.getView(), vk::ImageLayout::eGeneral),
    };

    auto imageWrite = [&](uint32_t binding, const vk::DescriptorImageInfo& info) {
        return vk::WriteDescriptorSet{}
            .setDstSet(descriptorSet.get())
            .setDstBinding(binding)
            .setDescriptorType(vk::DescriptorType::eStorageImage)
            .setDescriptorCount(1)
            .setImageInfo(info);
    };

    std::array<vk::WriteDescriptorSet, 6> writes {
        imageWrite(1, imageInfos[0]),
        imageWrite(2, imageInfos[1]),
        imageWrite(3, imageInfos[2]),
        imageWrite(4, imageInfos[3]),
        imageWrite(5, imageInfos[4]),
        imageWrite(6, imageInfos[5]),
    };
    context.getDevice().updateDescriptorSets(writes, {});
}

void WavefrontRaytracer::writeCameraRayLutDescriptor()
{
    if (!cameraRayLutBuffer.getBuffer()) {
        RayLutEntry dummy{};
        dummy.direction = float3(0.0f, 0.0f, 1.0f);
        cameraRayLutBuffer = Buffer(context, Buffer::Type::Storage, sizeof(RayLutEntry), &dummy);
    }

    vk::DescriptorBufferInfo info = cameraRayLutBuffer.getDescriptorInfo();
    vk::WriteDescriptorSet write{};
    write.setDstSet(descriptorSet.get()).setDstBinding(10)
         .setDescriptorType(vk::DescriptorType::eStorageBuffer)
         .setDescriptorCount(1).setBufferInfo(info);
    context.getDevice().updateDescriptorSets(write, {});
}

// ── updateMeshes ─────────────────────────────────────────────────────────────

void WavefrontRaytracer::updateMeshes()
{
    const auto& meshAssets = scene.getMeshAssets();
    std::vector<MeshAddresses> addrs;
    addrs.reserve(meshAssets.size());
    for (const auto& ma : meshAssets) {
        ma->updateMaterials();
        addrs.push_back(ma->getBufferAddresses());
    }
    if (addrs.empty()) {
        MeshAddresses dummy{};
        meshBuffer = Buffer{context, Buffer::Type::Storage, sizeof(MeshAddresses), &dummy};
    } else {
        meshBuffer = Buffer{context, Buffer::Type::Storage, sizeof(MeshAddresses) * addrs.size(), addrs.data()};
    }

    // Binding 7: meshes buffer
    vk::DescriptorBufferInfo info = meshBuffer.getDescriptorInfo();
    vk::WriteDescriptorSet write{};
    write.setDstSet(descriptorSet.get()).setDstBinding(7)
         .setDescriptorType(vk::DescriptorType::eStorageBuffer)
         .setDescriptorCount(1).setBufferInfo(info);
    context.getDevice().updateDescriptorSets(write, {});
}

// ── updateSceneSettings ───────────────────────────────────────────────────────

void WavefrontRaytracer::updateSceneSettings(const SceneSettings& ss)
{
    sceneSettingsBuffer = Buffer{context, Buffer::Type::Storage, sizeof(SceneSettings), &ss};
    maxShaderBounces = std::max({ss.renderSettings.diffuseBounces,
                                 ss.renderSettings.specularBounces,
                                 ss.renderSettings.transmissionBounces,
                                 1}) + 1;

    // Binding 8: sceneSettings
    {
        vk::DescriptorBufferInfo info = sceneSettingsBuffer.getDescriptorInfo();
        vk::WriteDescriptorSet write{};
        write.setDstSet(descriptorSet.get()).setDstBinding(8)
             .setDescriptorType(vk::DescriptorType::eStorageBuffer)
             .setDescriptorCount(1).setBufferInfo(info);
        context.getDevice().updateDescriptorSets(write, {});
    }
}

bool WavefrontRaytracer::prepareCameraRayLut(PushData& pc, const SceneSettings& sceneSettings)
{
    pc.rayLutEnabled = 0;
    currentCameraType = sceneSettings.camera.cameraType;
    if (sceneSettings.camera.cameraType != CAMERA_REALISTIC)
        return false;

    if (pc.pixelSizePercent != lastLutPixelSizePercent || sceneSettings.camera.cameraType != lastLutCameraType) {
        cameraRayLutValid = false;
        lastLutPixelSizePercent = pc.pixelSizePercent;
        lastLutCameraType = sceneSettings.camera.cameraType;
    }

    if (cameraRayLutValid) {
        pc.rayLutEnabled = 1;
        return false;
    }

    cameraRayLutNeedsGeneration = true;

    const CameraBase* activeCamera = scene.getActiveCamera();
    const std::string cacheFolder = activeCamera ? activeCamera->getRayLutCacheFolder() : std::string{};

    const size_t entryCount = static_cast<size_t>(width) * static_cast<size_t>(height);
    std::vector<RayLutEntry> entries(entryCount);
    for (RayLutEntry& entry : entries) {
        entry.direction = float3(0.0f, 0.0f, 1.0f);
        entry.originValid = 0.0f;
    }

    bool loadedFromFile = false;
    if (!cacheFolder.empty() && std::filesystem::is_directory(cacheFolder)) {
        RayLUTFileReader reader;
        for (const auto& dirEntry : std::filesystem::directory_iterator(cacheFolder)) {
            if (!dirEntry.is_regular_file() || dirEntry.path().extension() != ".raylut")
                continue;

            try {
                const RayLUT lut = reader.read(dirEntry.path());
                if (lut.rasterWidth != width || lut.rasterHeight != height || lut.empty())
                    continue;

                const float wavelength = lut.wavelengths.front();
                for (uint32_t y = 0; y < height; ++y) {
                    for (uint32_t x = 0; x < width; ++x) {
                        if (auto ray = lut.lookupInterpolated(static_cast<float>(x), static_cast<float>(y), wavelength))
                            entries[x + y * width] = *ray;
                    }
                }
                loadedFromFile = true;
                break;
            } catch (const std::exception&) {
                continue;
            }
        }
    }

    cameraRayLutBuffer = Buffer(context, Buffer::Type::Storage, sizeof(RayLutEntry) * entryCount,
                                loadedFromFile ? entries.data() : nullptr);
    writeCameraRayLutDescriptor();
    cameraRayLutValid = true;
    cameraRayLutNeedsGeneration = !loadedFromFile;
    pc.rayLutEnabled = 1;
    return true;
}

// ── Dispatch counts ───────────────────────────────────────────────────────────

void WavefrontRaytracer::updateDispatchCounts()
{
    const uint32_t w = std::max(1u, width);
    const uint32_t h = std::max(1u, height);
    groups2Dx = (w + uint32_t(GROUP_SIZE) - 1u) / uint32_t(GROUP_SIZE);
    groups2Dy = (h + uint32_t(GROUP_SIZE) - 1u) / uint32_t(GROUP_SIZE);
    extGroups  = (w * h + WavefrontGroupSize - 1u) / WavefrontGroupSize;
}

void WavefrontRaytracer::resize(const uint32_t newWidth, const uint32_t newHeight)
{
    if (newWidth == 0 || newHeight == 0 || (newWidth == width && newHeight == height))
        return;

    context.getDevice().waitIdle();
    Raytracer::resize(newWidth, newHeight);
    invalidateCameraRayLut();

    const vk::DeviceSize n = maxPixels();
    countersBuffer = Buffer(context, Buffer::Type::Storage, 4 * sizeof(int32_t));
    pathStatesBuffer = Buffer(context, Buffer::Type::Storage, n * 112);
    rayQueueBuffer = Buffer(context, Buffer::Type::Storage, n * 32);
    hitQueueBuffer = Buffer(context, Buffer::Type::Storage, n * 48);
    shadowQueueBuffer = Buffer(context, Buffer::Type::Storage, n * 2 * 48);

    writeWavefrontDescriptors();
    bindOutputImages();
    updateDispatchCounts();
}

// ── Render (inline dispatch, no prerecording) ─────────────────────────────────

void WavefrontRaytracer::render(const vk::CommandBuffer& cmd, const PushData& pc)
{
    cmd.pushConstants(pipelineLayout.get(), vk::ShaderStageFlagBits::eCompute,
                      0, sizeof(PushData), &pc);

    // Reset wavefront counters
    cmd.fillBuffer(countersBuffer.getBuffer(), 0, 4 * sizeof(int32_t), 0);

    // On frame 0, transition all output images from Undefined→General
    auto makeImageBarrier = [&](vk::Image img) {
        return vk::ImageMemoryBarrier{}
            .setOldLayout(vk::ImageLayout::eUndefined)
            .setNewLayout(vk::ImageLayout::eGeneral)
            .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setImage(img)
            .setSubresourceRange({vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1})
            .setSrcAccessMask({})
            .setDstAccessMask(vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
    };

    vk::MemoryBarrier fillBarrier{};
    fillBarrier.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite);
    fillBarrier.setDstAccessMask(vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);

    if (pc.frame == 0) {
        std::array imgBarriers{
            makeImageBarrier(outputColor.getImage()),
            makeImageBarrier(outputAlbedo.getImage()),
            makeImageBarrier(outputNormal.getImage()),
            makeImageBarrier(outputCrypto.getImage()),
            makeImageBarrier(outputPosition.getImage()),
            makeImageBarrier(outputMaterial.getImage()),
        };
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                            vk::PipelineStageFlagBits::eComputeShader,
                            {}, fillBarrier, {}, imgBarriers);
    } else {
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                            vk::PipelineStageFlagBits::eComputeShader,
                            {}, fillBarrier, {}, {});
    }

    bindAllDescriptorSets(cmd);

    if (currentCameraType == CAMERA_REALISTIC && pc.rayLutEnabled != 0 && cameraRayLutNeedsGeneration) {
        rayLutGenerator->writeDescriptors(sceneSettingsBuffer, cameraRayLutBuffer);
        rayLutGenerator->dispatch(cmd, width, height);
        computeBarrier(cmd);
        cameraRayLutNeedsGeneration = false;
        bindAllDescriptorSets(cmd);
    }

    // Generate: write primary rays into outputRayQueue, init path states
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines[Generate].get());
    cmd.dispatch(groups2Dx, groups2Dy, 1);
    computeBarrier(cmd);

    // Advance: promote outputRayQueue→inputRayQueue, reset counts
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines[Advance].get());
    cmd.dispatch(1, 1, 1);
    computeBarrier(cmd);

    for (int bounce = 0; bounce < maxShaderBounces; ++bounce) {
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines[Extend].get());
        cmd.dispatch(extGroups, 1, 1);
        computeBarrier(cmd);

        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines[Shade].get());
        cmd.dispatch(extGroups, 1, 1);
        computeBarrier(cmd);

        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines[Connect].get());
        cmd.dispatch(extGroups, 1, 1);
        computeBarrier(cmd);

        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines[Advance].get());
        cmd.dispatch(1, 1, 1);
        computeBarrier(cmd);
    }

    // Finalize: accumulate and write output images
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines[Finalize].get());
    cmd.dispatch(groups2Dx, groups2Dy, 1);
}

// ── Descriptor set binding ────────────────────────────────────────────────────

void WavefrontRaytracer::bindAllDescriptorSets(const vk::CommandBuffer& cmd) const
{
    std::array<vk::DescriptorSet, 2> sets{
        descriptorSet.get(),
        wavefrontDescSet.get()
    };
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                           pipelineLayout.get(), 0, sets, {});
}

// ── Barriers ─────────────────────────────────────────────────────────────────

void WavefrontRaytracer::computeBarrier(const vk::CommandBuffer& cmd) const
{
    vk::MemoryBarrier mb{};
    mb.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite);
    mb.setDstAccessMask(vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
    cmd.pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        {}, mb, {}, {});
}
