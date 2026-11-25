#pragma once

#include <cstdint>
#include <expected>
#include <span>

#include "DiffRenderer/Core/Image.h"
#include "DiffRenderer/Methods/Mesh/Shaders/MeshRasterizing.h"
#include "DiffRenderer/Methods/Mesh/SurfaceMesh.h"
#include "DiffRenderer/Methods/Mesh/TextureAtlas.h"
#include "DiffRenderer/Scenes/Dataset.h"
#include "NoorRHI/NoorRHI.h"

namespace DiffRenderer::Methods {

//! nvdiffrast with the split-sum shading of nvdiffrec: the hardware rasterizer writes triangle ids, compute passes shade, antialias and back-propagate.
class MeshRasterizer {
public:
    [[nodiscard]] std::expected<void, NoorRHI::Error> open(NoorRHI::Device& device, std::size_t maxPixels, NoorRHI::float3 background, bool antialiasing,
                                                           float nearPlane, float farPlane);
    void close();
    [[nodiscard]] std::expected<void, NoorRHI::Error> upload(const SurfaceMesh& mesh);
    [[nodiscard]] std::expected<void, NoorRHI::Error> download(SurfaceMesh& mesh) const;
    void setLight(const EnvironmentLight::Light& light) { arguments.light = light; }
    //! The seed of the uv jitter and the weights of the kd smoothness and the occlusion terms of nvdiffrec.
    void setRegularizers(std::uint32_t seed, float smoothWeight, float occlusionWeight);
    //! Shades with the bilinearly filtered texture from now on, until the next upload of a mesh.
    [[nodiscard]] std::expected<void, NoorRHI::Error> setTexture(const TextureAtlas& atlas, std::span<const float> values);
    [[nodiscard]] std::expected<void, NoorRHI::Error> downloadTexture(std::span<float> values) const;
    [[nodiscard]] std::expected<void, NoorRHI::Error> recordForward(NoorRHI::CommandList& list, const Scenes::Camera& camera);
    void recordBackward(NoorRHI::CommandList& list) const;
    void recordClamp(NoorRHI::CommandList& list) const;
    [[nodiscard]] std::expected<void, NoorRHI::Error> render(const Scenes::Camera& camera, Core::Image& image);

    NoorRHI::Ptr<float> getImage() const { return image.getPtr(); }
    NoorRHI::Ptr<float> getImageGradient() const { return imageGradient.getPtr(); }
    NoorRHI::Ptr<float> getPositions() const { return positions.getPtr(); }
    NoorRHI::Ptr<float> getPositionGradient() const { return positionGradient.getPtr(); }
    NoorRHI::Ptr<float> getVertexMaterials() const { return vertexMaterials.getPtr(); }
    NoorRHI::Ptr<float> getVertexMaterialGradient() const { return vertexMaterialGradient.getPtr(); }
    NoorRHI::Ptr<float> getTexels() const { return texels.getPtr(); }
    NoorRHI::Ptr<float> getTexelGradient() const { return texelGradient.getPtr(); }
    std::uint32_t getTexelValueCount() const { return arguments.textureWidth * arguments.textureHeight * MeshRasterizing::texelChannels; }
    std::uint32_t getVertexCount() const { return arguments.vertexCount; }
    std::uint32_t getTriangleCount() const { return arguments.triangleCount; }

private:
    [[nodiscard]] std::expected<void, NoorRHI::Error> openTargets(std::uint32_t width, std::uint32_t height);
    MeshRasterizing::Projection makeProjection(const Scenes::Camera& camera) const;

    NoorRHI::Device* device = nullptr;
    float nearPlane = 0.0f;
    float farPlane = 0.0f;
    NoorRHI::ComputePipeline clear;
    NoorRHI::ComputePipeline project;
    NoorRHI::ComputePipeline accumulateNormals;
    NoorRHI::ComputePipeline finishNormals;
    NoorRHI::GraphicsPipeline raster;
    NoorRHI::ComputePipeline resolve;
    NoorRHI::ComputePipeline shade;
    NoorRHI::ComputePipeline antialias;
    NoorRHI::ComputePipeline antialiasBackward;
    NoorRHI::ComputePipeline shadeBackward;
    NoorRHI::ComputePipeline normalBackward;
    NoorRHI::ComputePipeline faceNormalBackward;
    NoorRHI::ComputePipeline clampMaterials;
    NoorRHI::ComputePipeline gatherGradients;
    NoorRHI::ComputePipeline gatherTexelGradients;
    NoorRHI::Image ids;
    NoorRHI::Image depth;
    NoorRHI::Buffer<float> positions;
    NoorRHI::Buffer<float> vertexMaterials;
    NoorRHI::Buffer<NoorRHI::uint> triangles;
    NoorRHI::Buffer<int> opposites;
    NoorRHI::Buffer<NoorRHI::float4> clip;
    NoorRHI::Buffer<NoorRHI::Atomic<float>> clipGradient;
    NoorRHI::Buffer<NoorRHI::Atomic<float>> vertexMaterialAccumulator;
    NoorRHI::Buffer<float> positionGradient;
    NoorRHI::Buffer<float> vertexMaterialGradient;
    NoorRHI::Buffer<NoorRHI::Atomic<float>> normalSums;
    NoorRHI::Buffer<NoorRHI::Atomic<float>> tangentSums;
    NoorRHI::Buffer<float> normals;
    NoorRHI::Buffer<float> tangents;
    NoorRHI::Buffer<NoorRHI::Atomic<float>> normalGradient;
    NoorRHI::Buffer<float> normalSumGradient;
    NoorRHI::Buffer<NoorRHI::Atomic<float>> shadingPositionGradient;
    NoorRHI::Buffer<NoorRHI::float2> splitSum;
    NoorRHI::Buffer<NoorRHI::float2> uvs;
    NoorRHI::Buffer<NoorRHI::uint> uvTriangles;
    NoorRHI::Buffer<float> texels;
    NoorRHI::Buffer<NoorRHI::Atomic<float>> texelAccumulator;
    NoorRHI::Buffer<float> texelGradient;
    NoorRHI::Buffer<NoorRHI::float4> rasterBuffer;
    NoorRHI::Buffer<float> shaded;
    NoorRHI::Buffer<float> image;
    NoorRHI::Buffer<float> imageGradient;
    NoorRHI::Buffer<float> shadedGradient;
    NoorRHI::Readback<float> imageReadback;
    MeshRasterizing::Arguments arguments{};
};

}
