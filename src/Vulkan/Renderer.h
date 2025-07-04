#pragma once

#include <memory>

#include "Accel.h"
#include "Buffer.h"
#include "Context.h"
#include "Texture.h"
#include "Camera/PerspectiveCamera.h"
#include "Mesh/MeshAsset.h"
#include "Shaders/SharedStructs.h"

class Renderer {
public:
    Context& context;
    uint32_t width, height;
    bool dirty = true;

    vk::UniqueSwapchainKHR swapchain;
    std::vector<vk::Image> swapchainImages;
    vk::UniqueSemaphore imageAcquiredSemaphore;

    Image inputImage;
    Buffer texturesBuffer;
    Buffer meshBuffer;
    Buffer instancesBuffer;
    Buffer raygenSBT;
    Buffer missSBT;
    Buffer hitSBT;
    Accel tlas;

    vk::UniqueDescriptorSetLayout descSetLayout;
    vk::UniqueDescriptorSet descriptorSet;

    vk::UniquePipeline pipeline;
    vk::UniquePipelineLayout pipelineLayout;
    std::vector<vk::UniqueShaderModule> shaderModules;

    std::vector<vk::UniqueCommandBuffer> commandBuffers;

    std::vector<Texture> textures;
    std::vector<std::string> textureNames;
    std::vector<vk::DescriptorImageInfo> textureImageInfos;

    std::vector<std::shared_ptr<MeshAsset>> meshAssets;
    std::vector<std::unique_ptr<SceneObject>> sceneObjects;
    PerspectiveCamera* activeCamera = nullptr;

    vk::StridedDeviceAddressRegionKHR raygenRegion;
    vk::StridedDeviceAddressRegionKHR missRegion;
    vk::StridedDeviceAddressRegionKHR hitRegion;

    std::shared_ptr<MeshAsset> cameraGizmoAsset;
    const std::shared_ptr<MeshAsset>& getCameraGizmoAsset() const { return cameraGizmoAsset; }
    
    template <class T>
    void updateStorageBuffer(uint32_t binding, const std::vector<T>& data, Buffer& buffer);

    void updateTextureDescriptors(const std::vector<Texture>& textures);

    void rebuildTLAS();

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

    ~Renderer();
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;
    Renderer(Renderer&&) = delete;
    Renderer& operator=(Renderer&&) = delete;

    void render(uint32_t imageIndex, const PushConstants& pushConstants);

    const vk::CommandBuffer& getCommandBuffer(uint32_t imageIndex) const;
    const vk::SwapchainKHR& getSwapChain() const;
    const std::vector<vk::Image>& getSwapchainImages() const;

    void add(Texture&& element);

    std::shared_ptr<MeshAsset> get(const std::string& name) const;

    void markDirty() { dirty = true; };
    void resetDirty() { dirty = false; };
    bool getDirty() const { return dirty; }

    const std::vector<std::string>& getTextureNames() const { return textureNames; }
};