#include "MeshAsset.h"
#include "Utils.h"
#include "Vulkan/Renderer.h"

std::shared_ptr<MeshAsset> MeshAsset::CreateCube(Context& context, const std::string& name) {
    std::vector<Vertex> vertices = {
        {{-0.5f, -0.5f, -0.5f}, 0, { 0,  0, -1}, 0, {0, 0}, 0, 0},
        {{ 0.5f, -0.5f, -0.5f}, 0, { 0,  0, -1}, 0, {1, 0}, 0, 0},
        {{ 0.5f,  0.5f, -0.5f}, 0, { 0,  0, -1}, 0, {1, 1}, 0, 0},
        {{-0.5f,  0.5f, -0.5f}, 0, { 0,  0, -1}, 0, {0, 1}, 0, 0},
        {{-0.5f, -0.5f,  0.5f}, 0, { 0,  0,  1}, 0, {0, 0}, 0, 0},
        {{ 0.5f, -0.5f,  0.5f}, 0, { 0,  0,  1}, 0, {1, 0}, 0, 0},
        {{ 0.5f,  0.5f,  0.5f}, 0, { 0,  0,  1}, 0, {1, 1}, 0, 0},
        {{-0.5f,  0.5f,  0.5f}, 0, { 0,  0,  1}, 0, {0, 1}, 0, 0},
    };

    std::vector<uint32_t> indices = {
        // Front face
        0, 1, 2, 2, 3, 0,
        // Back face
        4, 5, 6, 6, 7, 4,
        // Left face
        0, 4, 7, 7, 3, 0,
        // Right face
        1, 5, 6, 6, 2, 1,
        // Top face
        3, 2, 6, 6, 7, 3,
        // Bottom face
        0, 1, 5, 5, 4, 0
    };

    std::vector<Face> faces;
    for (size_t i = 0; i < indices.size(); i += 3) {
        Face face{};
        face.materialIndex = 0;
        faces.push_back(face);
    }

    return std::make_shared<MeshAsset>(context, name, std::move(vertices), std::move(indices), std::move(faces));
}

std::shared_ptr<MeshAsset> MeshAsset::CreatePlane(Context& context, const std::string& name) {
    std::vector<Vertex> vertices = {
        {{-0.5f, 0.0f, -0.5f}, 0, {0, 1, 0}, 0, {0, 0}, 0, 0},
        {{ 0.5f, 0.0f, -0.5f}, 0, {0, 1, 0}, 0, {1, 0}, 0, 0},
        {{ 0.5f, 0.0f,  0.5f}, 0, {0, 1, 0}, 0, {1, 1}, 0, 0},
        {{-0.5f, 0.0f,  0.5f}, 0, {0, 1, 0}, 0, {0, 1}, 0, 0},
    };

    std::vector<uint32_t> indices = {
        0, 1, 2,
        2, 3, 0,
    };

    std::vector<Face> faces;
    for (size_t i = 0; i < indices.size(); i += 3) {
        Face face{};
        face.materialIndex = 0;
        faces.push_back(face);
    }

    return std::make_shared<MeshAsset>(context, name, std::move(vertices), std::move(indices), std::move(faces));
}

std::shared_ptr<MeshAsset> MeshAsset::CreateSphere(Context& context, const std::string& name, uint32_t latSeg, uint32_t lonSeg) {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<Face> faces;

    for (uint32_t lat = 0; lat <= latSeg; ++lat) {
        float theta = lat * float(M_PI) / latSeg;
        float sinTheta = sin(theta);
        float cosTheta = cos(theta);

        for (uint32_t lon = 0; lon <= lonSeg; ++lon) {
            float phi = lon * 2.0f * float(M_PI) / lonSeg;
            float sinPhi = sin(phi);
            float cosPhi = cos(phi);

            float x = cosPhi * sinTheta;
            float y = cosTheta;
            float z = sinPhi * sinTheta;

            vec3 normal;
            if (lat == 0) {
                // Top pole normal straight up
                normal = {0.0f, 1.0f, 0.0f};
            } else if (lat == latSeg) {
                // Bottom pole normal straight down
                normal = {0.0f, -1.0f, 0.0f};
            } else {
                normal = {x, y, z};
            }

            vertices.push_back({
                {x * 0.5f, y * 0.5f, z * 0.5f},   // position
                0,                                // _pad0
                normal,                           // fixed normal
                0,                                // _pad1
                {lon / float(lonSeg), lat / float(latSeg)}, // uv
                0, 0                        // _pad2, _pad3
            });
        }
    }

    for (uint32_t lat = 0; lat < latSeg; ++lat) {
        for (uint32_t lon = 0; lon < lonSeg; ++lon) {
            uint32_t first = lat * (lonSeg + 1) + lon;
            uint32_t second = first + lonSeg + 1;

            indices.push_back(first);
            indices.push_back(second);
            indices.push_back(first + 1);
            {
                Face face{};
                face.materialIndex = 0;
                faces.push_back(face);
            }

            indices.push_back(second);
            indices.push_back(second + 1);
            indices.push_back(first + 1);
            {
                Face face{};
                face.materialIndex = 0;
                faces.push_back(face);
            }
        }
    }
    return std::make_shared<MeshAsset>(context, name, std::move(vertices), std::move(indices), std::move(faces));
}

std::shared_ptr<MeshAsset> MeshAsset::CreateFromObj(Context& context, Renderer& renderer, const std::string& objFilePath)
{
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<Face> faces;

    Utils::loadObj(context, renderer, objFilePath, vertices, indices, faces);

    return std::make_shared<MeshAsset>(context, objFilePath, std::move(vertices), std::move(indices), std::move(faces));
}

MeshAsset::MeshAsset(Context& context, const std::string& name, const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices, const std::vector<Face>& faces)
: path(name) {
    
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
