#include "RtxRaytracer.h"
#include "Globals.h"
#include "Scene/MeshInstance.h"

static constexpr unsigned char GenSpv[] = {
#embed "../Shaders/Wavefront/GenerateRtx.spv"
};
static constexpr unsigned char ExtSpv[] = {
#embed "../Shaders/Wavefront/ExtendRtx.spv"
};
static constexpr unsigned char ShdSpv[] = {
#embed "../Shaders/Wavefront/ShadeRtx.spv"
};
static constexpr unsigned char ConSpv[] = {
#embed "../Shaders/Wavefront/ConnectRtx.spv"
};
static constexpr unsigned char FinSpv[] = {
#embed "../Shaders/Wavefront/FinalizeRtx.spv"
};
static constexpr unsigned char AdvSpv[] = {
#embed "../Shaders/Wavefront/AdvanceRtx.spv"
};

void RtxRaytracer::createTlasSet()
{
    const vk::DescriptorSetLayoutBinding tlasBinding{
        0, vk::DescriptorType::eAccelerationStructureKHR, 1, vk::ShaderStageFlagBits::eCompute
    };
    vk::DescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.setBindings(tlasBinding);
    tlasSetLayout = context.getDevice().createDescriptorSetLayoutUnique(layoutInfo);

    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo.setDescriptorPool(context.getDescriptorPool());
    allocInfo.setSetLayouts(tlasSetLayout.get());
    tlasDescSet = std::move(context.getDevice().allocateDescriptorSetsUnique(allocInfo).front());
}

void RtxRaytracer::buildPipelineLayout()
{
    // 3 descriptor sets: scene resources, wavefront queues, TLAS
    std::array<vk::DescriptorSetLayout, 3> setLayouts {
        descSetLayout.get(), wavefrontSetLayout.get(), tlasSetLayout.get()
    };

    vk::PushConstantRange pcRange{};
    pcRange.setStageFlags(vk::ShaderStageFlagBits::eCompute);
    pcRange.setSize(sizeof(PushData));

    vk::PipelineLayoutCreateInfo info{};
    info.setSetLayouts(setLayouts);
    info.setPushConstantRanges(pcRange);
    pipelineLayout = context.getDevice().createPipelineLayoutUnique(info);
}

void RtxRaytracer::bindAllDescriptorSets(const vk::CommandBuffer& cmd) const
{
    std::array<vk::DescriptorSet, 3> sets{
        descriptorSet.get(),
        wavefrontDescSet.get(),
        tlasDescSet.get()
    };
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                           pipelineLayout.get(), 0, sets, {});
}

RtxRaytracer::RtxRaytracer(Scene& scene, uint32_t width, uint32_t height)
    : WavefrontRaytracer(scene, width, height)
{
    // Set 0: instances(0) + outputColor(1) + outputAlbedo(2) + outputNormal(3) + outputCrypto(4)
    //        + outputPosition(5) + outputMaterial(6) + meshes(7) + sceneSettings(8)
    //        + sceneLights(9) + rayLUT(10) + textures(11, LAST variable-count)
    // Set 2: TLAS at binding 0 (separate set, NOT in set 0)
    createSceneDescriptorSet({
        {0,  vk::DescriptorType::eStorageBuffer,        1,            vk::ShaderStageFlagBits::eCompute},
        {1,  vk::DescriptorType::eStorageImage,         1,            vk::ShaderStageFlagBits::eCompute},
        {2,  vk::DescriptorType::eStorageImage,         1,            vk::ShaderStageFlagBits::eCompute},
        {3,  vk::DescriptorType::eStorageImage,         1,            vk::ShaderStageFlagBits::eCompute},
        {4,  vk::DescriptorType::eStorageImage,         1,            vk::ShaderStageFlagBits::eCompute},
        {5,  vk::DescriptorType::eStorageImage,         1,            vk::ShaderStageFlagBits::eCompute},
        {6,  vk::DescriptorType::eStorageImage,         1,            vk::ShaderStageFlagBits::eCompute},
        {7,  vk::DescriptorType::eStorageBuffer,        1,            vk::ShaderStageFlagBits::eCompute},
        {8,  vk::DescriptorType::eStorageBuffer,        1,            vk::ShaderStageFlagBits::eCompute},
        {9,  vk::DescriptorType::eStorageBuffer,        1,            vk::ShaderStageFlagBits::eCompute},
        {10, vk::DescriptorType::eStorageBuffer,        1,            vk::ShaderStageFlagBits::eCompute},
        {11, vk::DescriptorType::eCombinedImageSampler, MAX_TEXTURES, vk::ShaderStageFlagBits::eCompute},
    });

    createWavefrontSet();
    createTlasSet();
    buildPipelineLayout();

    buildPipelines({{
        {GenSpv, sizeof(GenSpv)},
        {ExtSpv, sizeof(ExtSpv)},
        {ShdSpv, sizeof(ShdSpv)},
        {ConSpv, sizeof(ConSpv)},
        {FinSpv, sizeof(FinSpv)},
        {AdvSpv, sizeof(AdvSpv)},
    }});

    writeWavefrontDescriptors();
    bindOutputImages();
    writeCameraRayLutDescriptor();

    // Initialize sceneSettings with defaults
    SceneSettings dummy{};
    sceneSettingsBuffer = Buffer{context, Buffer::Type::Storage, sizeof(SceneSettings), &dummy};
    {
        vk::DescriptorBufferInfo info = sceneSettingsBuffer.getDescriptorInfo();
        vk::WriteDescriptorSet write{};
        write.setDstSet(descriptorSet.get()).setDstBinding(8)
             .setDescriptorType(vk::DescriptorType::eStorageBuffer)
             .setDescriptorCount(1).setBufferInfo(info);
        context.getDevice().updateDescriptorSets(write, {});
    }

    // Initialize sceneLights with a dummy (no lights)
    LightGpu dummyLight{};
    sceneLightsBuffer = Buffer{context, Buffer::Type::Storage, sizeof(LightGpu), &dummyLight};
    {
        vk::DescriptorBufferInfo info = sceneLightsBuffer.getDescriptorInfo();
        vk::WriteDescriptorSet write{};
        write.setDstSet(descriptorSet.get()).setDstBinding(9)
             .setDescriptorType(vk::DescriptorType::eStorageBuffer)
             .setDescriptorCount(1).setBufferInfo(info);
        context.getDevice().updateDescriptorSets(write, {});
    }
}

void RtxRaytracer::updateTLAS()
{
    const auto& meshInstances = scene.getMeshInstances();

    std::vector<vk::AccelerationStructureInstanceKHR> accelInstances;
    std::vector<Instance> gpuInstances;
    accelInstances.reserve(meshInstances.size());
    gpuInstances.reserve(meshInstances.size());
    for (const auto* mi : meshInstances) {
        if (!mi)
            continue;

        const uint32_t instanceIndex = static_cast<uint32_t>(gpuInstances.size());

        auto accelInstance = mi->getInstanceData();
        accelInstance.setInstanceCustomIndex(instanceIndex);
        accelInstances.push_back(accelInstance);

        const mat4 T  = mi->getWorldTransform().getMatrix();
        const mat4 IT = inverse(T);
        gpuInstances.push_back({T, IT, transpose(IT), mi->getMeshAsset().getMeshIndex()});
    }

    if (accelInstances.empty()) {
        vk::AccelerationStructureInstanceKHR emptyInst{};
        instancesBuffer = Buffer{context, Buffer::Type::AccelInput,
                                 sizeof(vk::AccelerationStructureInstanceKHR), &emptyInst};
    } else {
        instancesBuffer = Buffer{context, Buffer::Type::AccelInput,
                                 sizeof(vk::AccelerationStructureInstanceKHR) * accelInstances.size(),
                                 accelInstances.data()};
    }

    vk::AccelerationStructureGeometryInstancesDataKHR instData{};
    instData.setArrayOfPointers(false);
    instData.setData(instancesBuffer.getDeviceAddress());

    vk::AccelerationStructureGeometryKHR geom{};
    geom.setGeometryType(vk::GeometryTypeKHR::eInstances);
    geom.setGeometry({instData});
    geom.setFlags(vk::GeometryFlagBitsKHR::eOpaque);

    tlas.build(context, geom, static_cast<uint32_t>(accelInstances.size()),
               vk::AccelerationStructureTypeKHR::eTopLevel);

    // Write TLAS to set 2 binding 0 (NOT set 0)
    vk::WriteDescriptorSetAccelerationStructureKHR accelInfo{};
    accelInfo.setAccelerationStructureCount(1);
    accelInfo.setPAccelerationStructures(&tlas.getAccelerationStructure());

    vk::WriteDescriptorSet accelWrite{};
    accelWrite.setDstSet(tlasDescSet.get()).setDstBinding(0)
              .setDescriptorType(vk::DescriptorType::eAccelerationStructureKHR)
              .setDescriptorCount(1).setPNext(&accelInfo);
    context.getDevice().updateDescriptorSets(accelWrite, {});

    // Also write the per-instance data buffer to set 0 binding 0.
    // RayQuery::CommittedInstanceID returns the TLAS custom index, so the
    // custom index must match this buffer rather than the shared mesh index.
    if (gpuInstances.empty()) {
        Instance dummy{}; dummy.meshId = UINT32_MAX;
        instancesBuffer = Buffer{context, Buffer::Type::Storage, sizeof(Instance), &dummy};
    } else {
        instancesBuffer = Buffer{context, Buffer::Type::Storage,
                                 sizeof(Instance) * gpuInstances.size(), gpuInstances.data()};
    }

    vk::DescriptorBufferInfo instInfo = instancesBuffer.getDescriptorInfo();
    vk::WriteDescriptorSet instWrite{};
    instWrite.setDstSet(descriptorSet.get()).setDstBinding(0)
             .setDescriptorType(vk::DescriptorType::eStorageBuffer)
             .setDescriptorCount(1).setBufferInfo(instInfo);
    context.getDevice().updateDescriptorSets(instWrite, {});
}

void RtxRaytracer::updateTextures()
{
    const auto& textures = scene.getTextures();
    if (textures.empty()) return;

    std::vector<vk::DescriptorImageInfo> imageInfos;
    imageInfos.reserve(textures.size());
    for (const auto& tex : textures) {
        vk::DescriptorImageInfo info{};
        info.setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
        info.setImageView(tex.getImage().getView());
        info.setSampler(tex.getSampler());
        imageInfos.push_back(info);
    }

    // Binding 11: textures (variable count, last binding)
    const vk::WriteDescriptorSet write{
        descriptorSet.get(), 11, 0,
        static_cast<uint32_t>(imageInfos.size()),
        vk::DescriptorType::eCombinedImageSampler,
        imageInfos.data()
    };
    context.getDevice().updateDescriptorSets(write, {});
}
