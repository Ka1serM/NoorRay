#pragma once

#include <memory>

#include "Accel.h"
#include "Buffer.h"
#include "Context.h"
#include "Texture.h"
#include "Camera/PerspectiveCamera.h"
#include "Mesh/MeshAsset.h"
#include "Shaders/SharedStructs.h"

template<class>
inline constexpr bool always_false = false;

class Renderer {

public:

    Image inputImage;

    bool dirty;

    uint32_t width, height;
    
    Context& context;

    vk::UniqueSwapchainKHR swapchain;
    std::vector<vk::Image> swapchainImages;
    
    std::vector<Texture> textures;
    std::vector<std::string> textureNames;
    Buffer texturesBuffer;
    std::vector<vk::DescriptorImageInfo> textureImageInfos;
        
    PerspectiveCamera* activeCamera = nullptr;  // non-owning pointer

    template <class T>
    void updateStorageBuffer(uint32_t binding, const std::vector<T>& data, Buffer& buffer);
    
    void updateTextureDescriptors(const std::vector<Texture>& textures);


    void rebuildTLAS();

    std::vector<std::shared_ptr<MeshAsset>> meshAssets;
    Buffer meshBuffer;
    
    std::vector<std::unique_ptr<SceneObject>> sceneObjects;
    Buffer instancesBuffer;

    void setActiveCamera(PerspectiveCamera* camera) {
        activeCamera = camera;
    }

    PerspectiveCamera* getActiveCamera() const {
        return activeCamera;
    }

    int add(std::unique_ptr<SceneObject> sceneObject);
    void rebuildMeshBuffer();

    void add(const std::shared_ptr<MeshAsset>& meshInstance);
    bool remove(const SceneObject* obj);
        
    Renderer(Context& context, uint32_t width, uint32_t height);
    
    void render(uint32_t imageIndex, const PushConstants& pushConstants);
    
    const vk::CommandBuffer& getCommandBuffer(uint32_t imageIndex) const;
    const vk::SwapchainKHR& getSwapChain() const;
    const std::vector<vk::Image>& getSwapchainImages() const;

    void add(Texture&& element);

    std::shared_ptr<MeshAsset> get(const std::string& name) const;
    
    vk::UniqueDescriptorSetLayout descSetLayout;
    vk::UniqueDescriptorSet descriptorSet;
    std::vector<vk::UniqueCommandBuffer> commandBuffers;

    std::vector<vk::UniqueShaderModule> shaderModules;
    vk::UniquePipelineLayout pipelineLayout;
    vk::UniquePipeline pipeline;
    vk::UniqueSemaphore imageAcquiredSemaphore;

    Buffer raygenSBT;
    Buffer missSBT;
    Buffer hitSBT;

    vk::StridedDeviceAddressRegionKHR raygenRegion;
    vk::StridedDeviceAddressRegionKHR missRegion;
    vk::StridedDeviceAddressRegionKHR hitRegion;

    Accel tlas;
    
    void markDirty() { dirty = true; };
    void resetDirty() { dirty = false; };
    bool getDirty() const { return dirty; }

    const std::vector<std::string>& getTextureNames() const { return textureNames; }
};
