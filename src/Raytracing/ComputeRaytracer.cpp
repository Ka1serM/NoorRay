#include "ComputeRaytracer.h"
#include "Globals.h"
#include "Scene/MeshInstance.h"

static constexpr unsigned char GenSpv[] = {
#embed "../Shaders/Wavefront/Generate.spv"
};
static constexpr unsigned char ExtSpv[] = {
#embed "../Shaders/Wavefront/Extend.spv"
};
static constexpr unsigned char ShdSpv[] = {
#embed "../Shaders/Wavefront/Shade.spv"
};
static constexpr unsigned char ConSpv[] = {
#embed "../Shaders/Wavefront/Connect.spv"
};
static constexpr unsigned char FinSpv[] = {
#embed "../Shaders/Wavefront/Finalize.spv"
};
static constexpr unsigned char AdvSpv[] = {
#embed "../Shaders/Wavefront/Advance.spv"
};

ComputeRaytracer::ComputeRaytracer(Scene& scene, uint32_t width, uint32_t height)
    : WavefrontRaytracer(scene, width, height)
{
    // Set 0: instances(0) + outputColor(1) + outputAlbedo(2) + outputNormal(3) + outputCrypto(4)
    //        + outputPosition(5) + outputMaterial(6) + meshes(7) + sceneSettings(8)
    //        + sceneLights(9) + rayLUT(10) + textures(11, LAST variable-count)
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

void ComputeRaytracer::updateTLAS()
{
    const auto& meshInstances = scene.getMeshInstances();
    std::vector<Instance> gpuInstances;
    gpuInstances.reserve(meshInstances.size());

    for (const auto* mi : meshInstances) {
        const mat4 T  = mi->getWorldTransform().getMatrix();
        const mat4 IT = inverse(T);
        gpuInstances.push_back({T, IT, transpose(IT), mi->getMeshAsset().getMeshIndex()});
    }

    if (gpuInstances.empty()) {
        Instance dummy{}; dummy.meshId = UINT32_MAX;
        instancesBuffer = Buffer{context, Buffer::Type::Storage, sizeof(Instance), &dummy};
    } else {
        instancesBuffer = Buffer{context, Buffer::Type::Storage,
                                 sizeof(Instance) * gpuInstances.size(), gpuInstances.data()};
    }

    vk::DescriptorBufferInfo info = instancesBuffer.getDescriptorInfo();
    vk::WriteDescriptorSet write{};
    write.setDstSet(descriptorSet.get()).setDstBinding(0)
         .setDescriptorType(vk::DescriptorType::eStorageBuffer)
         .setDescriptorCount(1).setBufferInfo(info);
    context.getDevice().updateDescriptorSets(write, {});
}

void ComputeRaytracer::updateTextures()
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
