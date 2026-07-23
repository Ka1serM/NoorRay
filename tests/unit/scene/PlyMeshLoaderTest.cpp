#include <catch2/catch_test_macros.hpp>

#include "Mesh/Assets/PlyMeshLoader.h"

TEST_CASE("PLY mesh loader triangulates polygon faces", "[ply]")
{
    const std::filesystem::path path =
        std::filesystem::path(PLY_TEST_DIR) / "quad_mesh.ply";
    REQUIRE(PlyMeshLoader::HasFaces(path));

    const PlyMeshData mesh = PlyMeshLoader::Load(path);
    REQUIRE(mesh.vertices.size() == 4);
    CHECK(mesh.indices == std::vector<uint32_t>{0, 1, 2, 0, 2, 3});
    for (const Vertex& vertex : mesh.vertices) {
        const glm::vec3 normalError = vertex.normal - glm::vec3(0.f, 0.f, 1.f);
        CHECK(glm::dot(normalError, normalError) < 1e-6f);
        CHECK(glm::dot(vertex.tangent, vertex.tangent) > .99f);
    }
}

TEST_CASE("PLY mesh detection rejects point-only PLY data", "[ply]")
{
    const std::filesystem::path path =
        std::filesystem::path(PLY_TEST_DIR) / "point_cloud.ply";
    CHECK_FALSE(PlyMeshLoader::HasFaces(path));
}
