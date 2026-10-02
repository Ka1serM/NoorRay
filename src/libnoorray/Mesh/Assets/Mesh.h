#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <noorrhi/noorrhi.hpp>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include "Shared/Mesh.h"
#include "Shared/SplineMesh.h"

class Scene;
class SplineMeshPass;

// One vertex's tangent basis as Unreal stores it (FPackedNormal): TangentX,
// then TangentZ, the normal, whose w is the bitangent sign. Each is SNORM8x4,
// x in the low byte. The bitangent is cross(normal, tangent) * sign.
struct TangentFrame
{
    TangentFrame() = default;
    TangentFrame(glm::vec3 tangent, glm::vec3 normal, float bitangentSign);

    glm::vec3 tangent() const;
    glm::vec3 normal() const;
    float bitangentSign() const;

    int8_t tangentX[4]{};
    int8_t tangentZ[4]{};
};
static_assert(sizeof(TangentFrame) == 8);

// A contiguous run of triangles drawn with one material slot, as Unreal's
// mesh sections are. A mesh's sections tile its triangles in order; several
// may use the same slot. Each is one BLAS geometry, and its shader-binding-
// table records select the hit groups of the material an instance puts in
// that slot.
struct MeshSection
{
    uint32_t slot{};
    uint32_t firstTriangle{};
    uint32_t triangleCount{};
    // Shadow rays pass the section when false, as Unreal's section flag does.
    bool castsShadow{true};
};

// For sources that assign a material slot per triangle: stably reorders the
// triangles of `indices` by slot and returns the sections that result.
std::vector<MeshSection> sortTrianglesBySlot(std::vector<uint32_t>& indices,
    std::span<const uint32_t> triangleSlots);

// The one section of a mesh drawn with a single material.
std::vector<MeshSection> singleSection(const std::vector<uint32_t>& indices);

// Vertex streams a producer builds, in the layout of nr::graphics::Mesh.
struct MeshStreams
{
    std::vector<glm::vec3> positions;
    std::vector<TangentFrame> tangents;
    // uvCount channels per vertex.
    std::vector<glm::vec2> uvs;
    uint32_t uvCount = 1;
    // Empty, or one FColor per vertex (nr::vertex_color).
    std::vector<uint32_t> colors;
    std::vector<uint32_t> indices;
    std::vector<MeshSection> sections;
};

// The streams a mesh draws, in the layout of nr::graphics::Mesh. They are
// borrowed and `owner` keeps them alive, so a memory-mapped file uploads
// without an intermediate copy.
struct MeshGeometry
{
    MeshGeometry() = default;
    // Owns the streams and computes their bounds.
    explicit MeshGeometry(MeshStreams streams);

    std::span<const glm::vec3> positions;
    std::span<const TangentFrame> tangents;
    std::span<const glm::vec2> uvs;
    uint32_t uvCount = 1;
    std::span<const uint32_t> colors;
    std::span<const uint32_t> indices;
    std::vector<MeshSection> sections;
    // Of the vertices as drawn, bent when `spline` is set.
    glm::vec3 boundsMin{};
    glm::vec3 boundsMax{};
    std::shared_ptr<const void> owner;
    // Bends the uploaded positions and tangent frames on the GPU.
    std::optional<nr::graphics::SplineMesh> spline;
};

// Geometry only: the materials belong to the instances that draw it. Each
// mesh owns its shader record, reached through the inherited `data` member.
// The Raytracer publishes a table of pointers to these records rather than
// copying the structs into one array, so a mesh that re-uploads does not
// force the whole table to be rebuilt - the same arrangement Material uses.
class Mesh : public noorrhi::Shared<nr::graphics::Mesh>
{
public:
    static Mesh CreateCube(Scene& scene, const std::string& name);
    static Mesh CreatePlane(Scene& scene, const std::string& name);
    static Mesh CreateSphere(Scene& scene, const std::string& name,
        uint32_t latitudeSegments = 64, uint32_t longitudeSegments = 64);
    static Mesh CreateDisk(Scene& scene, const std::string& name, uint32_t segments = 64);

    Mesh(Scene& scene, std::string name, MeshGeometry geometry);
    Mesh(Mesh&& other) noexcept;
    Mesh(const Mesh&) = delete;
    Mesh& operator=(const Mesh&) = delete;
    Mesh& operator=(Mesh&& other) = delete;
    ~Mesh() = default;

    const std::string& getName() const { return path; }
    std::string getType() const { return "Mesh Asset"; }
    const std::string& getPath() const { return path; }
    uint32_t getMeshIndex() const { return index; }
    void setMeshIndex(uint32_t newIndex) { index = newIndex; }
    Scene& getScene() const { return scene; }

    const MeshGeometry& getGeometry() const { return geometry; }
    const std::vector<MeshSection>& getSections() const { return geometry.sections; }
    uint32_t getVertexCount() const { return static_cast<uint32_t>(geometry.positions.size()); }
    // Animated/deforming geometry builds a fast-build, updateable BLAS;
    // static geometry favors fast tracing and omits BLAS update support.
    bool isAnimated() const { return animated; }
    void setAnimated(bool value);
    // How many materials an instance must supply: one past the highest slot
    // a section draws with.
    uint32_t getSlotCount() const { return slotCount; }

    void replaceGeometry(MeshGeometry value);

    // Uploads the streams when the geometry changed since the last upload,
    // bending them with `splineMeshPass` when the geometry has a spline;
    // scene publication calls this for every changed mesh.
    void upload(noorrhi::Device& device, const SplineMeshPass& splineMeshPass);
    // The BLAS whose sections have the given opacity (opaque sections skip
    // any-hit stages), built on first use and kept until the geometry
    // changes. Instances whose materials agree on opacity share it.
    const noorrhi::AccelerationStructure& blas(noorrhi::Device& device, const std::vector<bool>& opacity);
    void releaseGpu();

private:
    // Throws unless the streams agree on the vertex count and the sections
    // tile the triangles in order; counts the material slots.
    void validate();

    Scene& scene;
    std::string path;
    uint32_t index = ~0u;
    bool gpuDirty = true;
    bool animated = false;
    MeshGeometry geometry;
    uint32_t slotCount = 0;
    noorrhi::Buffer<glm::vec3> positionBuffer;
    noorrhi::Buffer<TangentFrame> tangentBuffer;
    noorrhi::Buffer<glm::vec2> uvBuffer;
    noorrhi::Buffer<uint32_t> colorBuffer;
    noorrhi::Buffer<uint32_t> indexBuffer;
    noorrhi::Buffer<nr::graphics::SectionRecord> sectionBuffer;
    std::vector<std::pair<std::vector<bool>, noorrhi::AccelerationStructure>> blases;
};
