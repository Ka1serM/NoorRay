#include "MeshAsset.h"

#include "Utils.h"
#include "Vulkan/Renderer.h"

MeshAsset::MeshAsset(Context& context, Renderer& renderer, const std::string& objFilePath)
: name(objFilePath) {

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<Face> faces;

    Utils::loadObj(context, renderer, objFilePath, vertices, indices, faces);

    // Upload mesh data to GPU
    vertexBuffer = Buffer{context, Buffer::Type::AccelInput, sizeof(Vertex) * vertices.size(), vertices.data()};
    indexBuffer = Buffer{context, Buffer::Type::AccelInput, sizeof(uint32_t) * indices.size(), indices.data()};
    faceBuffer = Buffer{context, Buffer::Type::AccelInput, sizeof(Face) * faces.size(), faces.data()};

    vk::AccelerationStructureGeometryTrianglesDataKHR triangleData{};
    triangleData.setVertexFormat(vk::Format::eR32G32B32Sfloat);
    triangleData.setVertexData(vertexBuffer.getDeviceAddress());
    triangleData.setVertexStride(sizeof(Vertex));
    triangleData.setMaxVertex(static_cast<uint32_t>(vertices.size()));
    triangleData.setIndexType(vk::IndexType::eUint32);
    triangleData.setIndexData(indexBuffer.getDeviceAddress());

    vk::AccelerationStructureGeometryKHR geometry{};
    geometry.setGeometryType(vk::GeometryTypeKHR::eTriangles);
    geometry.setGeometry({triangleData});
    geometry.setFlags(vk::GeometryFlagBitsKHR::eOpaque);

    // Create bottom-level acceleration structure (BLAS)
    blas.build(context, geometry, static_cast<uint32_t>(indices.size() / 3), vk::AccelerationStructureTypeKHR::eBottomLevel);

    renderer.add(getBufferAddresses());
}

uint64_t MeshAsset::getBlasAddress() const {
    return blas.buffer.getDeviceAddress();
}

MeshAddresses MeshAsset::getBufferAddresses() const {
    return MeshAddresses{
        vertexBuffer.getDeviceAddress(),
        indexBuffer.getDeviceAddress(),
        faceBuffer.getDeviceAddress(),
    };
}

uint32_t MeshAsset::getMeshIndex() const {
    return index;
}

void MeshAsset::setMeshIndex(uint32_t newIndex) {
    index = newIndex;
}
