#include "mesh.h"

#include "instancer.h"
#include "material.h"
#include "renderParam.h"

#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3d.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/imaging/hd/changeTracker.h>
#include <pxr/imaging/hd/geomSubset.h>
#include <pxr/imaging/hd/renderIndex.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/imaging/hd/tokens.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <vector>

#include <glm/geometric.hpp>
#include <glm/vec4.hpp>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include "Geometry/Mesh/Assets/GaussianAsset.h"
#include "Geometry/Mesh/Assets/MeshAsset.h"
#include "Materials/MaterialX/MaterialXDocument.h"
#include "Geometry/Mesh/Transform.h"
#include "Scene/Objects/GaussianInstance.h"
#include "Scene/Objects/MeshInstance.h"
#include "Scene/Scene.h"

#include <pxr/base/tf/diagnostic.h>
#include <exception>

PXR_NAMESPACE_OPEN_SCOPE

namespace
{
const TfToken StToken("st");
constexpr size_t ParallelTriangleThreshold = 64 * 1024;
constexpr size_t ParallelTriangleGrainSize = 8 * 1024;

glm::mat4 ToGlm(const GfMatrix4d& value)
{
    glm::mat4 result(1.0f);
    for (int row = 0; row < 4; ++row)
        for (int column = 0; column < 4; ++column)
            result[column][row] = static_cast<float>(value[column][row]);
    return result;
}

std::vector<glm::vec3> Vec3Values(const VtValue& value)
{
    std::vector<glm::vec3> result;
    if (value.IsHolding<VtVec3fArray>()) {
        const VtVec3fArray& values = value.UncheckedGet<VtVec3fArray>();
        result.reserve(values.size());
        for (const GfVec3f& v : values)
            result.emplace_back(v[0], v[1], v[2]);
    } else if (value.IsHolding<VtVec3dArray>()) {
        const VtVec3dArray& values = value.UncheckedGet<VtVec3dArray>();
        result.reserve(values.size());
        for (const GfVec3d& v : values)
            result.emplace_back(v[0], v[1], v[2]);
    }
    return result;
}

// Keep Blender/USD UVs in their authored convention. Blender's live image
// buffers are already laid out with the V origin expected by the renderer's
// normalized texture sampler; changing V here also changes the tangent frame
// used by MaterialX normal maps and mirrors both color and normal textures.
std::vector<glm::vec2> Vec2Values(const VtValue& value)
{
    std::vector<glm::vec2> result;
    if (value.IsHolding<VtVec2fArray>()) {
        const VtVec2fArray& values = value.UncheckedGet<VtVec2fArray>();
        result.reserve(values.size());
        for (const GfVec2f& v : values)
            result.emplace_back(v[0], v[1]);
    }
    return result;
}

std::vector<float> FloatValues(const VtValue& value)
{
    std::vector<float> result;
    if (value.IsHolding<VtFloatArray>()) {
        const VtFloatArray& values = value.UncheckedGet<VtFloatArray>();
        result.reserve(values.size());
        for (const float v : values)
            result.push_back(v);
    } else if (value.IsHolding<VtDoubleArray>()) {
        const VtDoubleArray& values = value.UncheckedGet<VtDoubleArray>();
        result.reserve(values.size());
        for (const double v : values)
            result.push_back(static_cast<float>(v));
    }
    return result;
}

struct MeshPrimvars
{
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> uvs;
    std::vector<glm::vec3> colors;
    std::vector<float> opacities;
    HdInterpolation normalInterpolation{HdInterpolationConstant};
    HdInterpolation uvInterpolation{HdInterpolationConstant};
    HdInterpolation colorInterpolation{HdInterpolationConstant};
    HdInterpolation opacityInterpolation{HdInterpolationConstant};
};

MeshPrimvars ReadMeshPrimvars(
    HdSceneDelegate* delegate, const SdfPath& id,
    const VtValue* prefetchedSt)
{
    MeshPrimvars result;
    bool hasNormals = false;
    bool hasUvs = false;
    bool hasColors = false;
    bool hasOpacities = false;

    // GetPrimvarDescriptors returns a vector by value. Sweep each interpolation
    // once and record the four attributes we consume instead of asking Hydra
    // for the same descriptor vectors once per attribute.
    for (const HdInterpolation interpolation : {
             HdInterpolationConstant, HdInterpolationUniform,
             HdInterpolationVarying, HdInterpolationVertex,
             HdInterpolationFaceVarying}) {
        for (const HdPrimvarDescriptor& descriptor :
             delegate->GetPrimvarDescriptors(id, interpolation)) {
            if (descriptor.name == HdTokens->normals) {
                hasNormals = true;
                result.normalInterpolation = interpolation;
            } else if (descriptor.name == StToken) {
                hasUvs = true;
                result.uvInterpolation = interpolation;
            } else if (descriptor.name == HdTokens->displayColor) {
                hasColors = true;
                result.colorInterpolation = interpolation;
            } else if (descriptor.name == HdTokens->displayOpacity) {
                hasOpacities = true;
                result.opacityInterpolation = interpolation;
            }
        }
    }

    if (hasNormals)
        result.normals = Vec3Values(delegate->Get(id, HdTokens->normals));
    if (hasUvs) {
        result.uvs = Vec2Values(
            prefetchedSt != nullptr ? *prefetchedSt : delegate->Get(id, StToken));
    }
    if (hasColors)
        result.colors = Vec3Values(delegate->Get(id, HdTokens->displayColor));
    if (hasOpacities) {
        result.opacities =
            FloatValues(delegate->Get(id, HdTokens->displayOpacity));
    }
    return result;
}

template<typename T>
T PrimvarAt(
    const std::vector<T>& values, const HdInterpolation interpolation,
    const int pointIndex, const size_t cornerIndex, const size_t faceIndex,
    const T fallback)
{
    size_t index = 0;
    if (interpolation == HdInterpolationFaceVarying)
        index = cornerIndex;
    else if (interpolation == HdInterpolationVertex
        || interpolation == HdInterpolationVarying)
        index = pointIndex < 0 ? values.size() : static_cast<size_t>(pointIndex);
    else if (interpolation == HdInterpolationUniform)
        index = faceIndex;
    return index < values.size() ? values[index] : fallback;
}

glm::vec3 NormalizeOr(const glm::vec3& value, const glm::vec3& fallback)
{
    const float lengthSquared = glm::dot(value, value);
    return lengthSquared > 1e-20f
        ? value * glm::inversesqrt(lengthSquared)
        : fallback;
}

void PopulateTriangle(
    const std::vector<glm::vec3>& positions, const MeshPrimvars& primvars,
    const std::array<int, 3>& pointIndices,
    const std::array<size_t, 3>& cornerIndices, const size_t faceIndex,
    Vertex* triangle)
{
    const glm::vec3 edge1 =
        positions[pointIndices[1]] - positions[pointIndices[0]];
    const glm::vec3 edge2 =
        positions[pointIndices[2]] - positions[pointIndices[0]];
    const glm::vec3 geometricNormal = NormalizeOr(
        glm::cross(edge1, edge2), glm::vec3(0.0f, 0.0f, 1.0f));

    for (int c = 0; c < 3; ++c) {
        const int pointIndex = pointIndices[c];
        const size_t cornerIndex = cornerIndices[c];
        Vertex& vertex = triangle[c];
        vertex.position = positions[pointIndex];
        vertex.normal = NormalizeOr(
            PrimvarAt(
                primvars.normals, primvars.normalInterpolation, pointIndex,
                cornerIndex, faceIndex, geometricNormal),
            geometricNormal);
        vertex.uv = PrimvarAt(
            primvars.uvs, primvars.uvInterpolation, pointIndex, cornerIndex,
            faceIndex, glm::vec2(0.0f));
        const glm::vec3 color = PrimvarAt(
            primvars.colors, primvars.colorInterpolation, pointIndex,
            cornerIndex, faceIndex, glm::vec3(1.0f));
        const float opacity = PrimvarAt(
            primvars.opacities, primvars.opacityInterpolation, pointIndex,
            cornerIndex, faceIndex, 1.0f);
        vertex.color =
            nr::vertex_color::packLinear(glm::vec4(color, opacity));
        vertex.tangentSign = 1.0f;
    }

    const glm::vec2 duv1 = triangle[1].uv - triangle[0].uv;
    const glm::vec2 duv2 = triangle[2].uv - triangle[0].uv;
    const float determinant = duv1.x * duv2.y - duv1.y * duv2.x;
    const bool hasUvFrame = std::abs(determinant) > 1e-8f;
    const glm::vec3 tangentDerivative = hasUvFrame
        ? (edge1 * duv2.y - edge2 * duv1.y) / determinant
        : edge1;
    const glm::vec3 bitangentDerivative = hasUvFrame
        ? (edge2 * duv1.x - edge1 * duv2.x) / determinant
        : glm::cross(geometricNormal, tangentDerivative);
    for (int c = 0; c < 3; ++c) {
        Vertex& vertex = triangle[c];
        const glm::vec3 normal = vertex.normal;
        glm::vec3 tangent =
            tangentDerivative - normal * glm::dot(normal, tangentDerivative);
        if (glm::dot(tangent, tangent) <= 1e-12f) {
            const glm::vec3 axis = std::abs(normal.x) < 0.9f
                ? glm::vec3(1.0f, 0.0f, 0.0f)
                : glm::vec3(0.0f, 1.0f, 0.0f);
            tangent = glm::cross(axis, normal);
        }
        vertex.tangent = NormalizeOr(tangent, glm::vec3(1.0f, 0.0f, 0.0f));
        vertex.tangentSign =
            glm::dot(glm::cross(normal, vertex.tangent), bitangentDerivative)
                < 0.0f
            ? -1.0f
            : 1.0f;
    }
}

// Maps each face index to a local material slot: 0 is the Rprim's own
// primary material (GetMaterialId()), and 1+ are the distinct materialIds
// referenced by the mesh's HdGeomSubsets (HdGeomSubset::TypeFaceSet only --
// NoorRay's material assignment is per-triangle, matching this type exactly),
// in first-encountered order. `materialSlotPaths` receives the SdfPath for
// slot 0 (always requestedMaterialId, even if no subset uses it) and each
// subsequent slot in the same order, for the caller to bind via
// MeshAsset::setMaterial(slot, ...). A face not covered by any subset stays
// slot 0, the common case for single-material meshes (unchanged behavior)
// and for DCCs that, like Blender's Hydra export, already split a
// multi-material mesh into one Rprim per material rather than using
// GeomSubsets on one Rprim.
std::vector<int> BuildFaceMaterialSlots(
    const HdMeshTopology& topology, const SdfPath& requestedMaterialId,
    const size_t faceCount, std::vector<SdfPath>& materialSlotPaths)
{
    materialSlotPaths.assign(1, requestedMaterialId);
    std::vector<int> faceSlots(faceCount, 0);
    for (const HdGeomSubset& subset : topology.GetGeomSubsets()) {
        if (subset.type != HdGeomSubset::TypeFaceSet
            || subset.materialId.IsEmpty()
            || subset.materialId == requestedMaterialId)
            continue;
        const auto existing = std::ranges::find(materialSlotPaths, subset.materialId);
        const int slot = existing != materialSlotPaths.end()
            ? static_cast<int>(existing - materialSlotPaths.begin())
            : static_cast<int>(materialSlotPaths.size());
        if (existing == materialSlotPaths.end())
            materialSlotPaths.push_back(subset.materialId);
        for (const int faceIndex : subset.indices)
            if (faceIndex >= 0 && static_cast<size_t>(faceIndex) < faceCount)
                faceSlots[faceIndex] = slot;
    }
    return faceSlots;
}

void BuildTriangleMesh(
    HdSceneDelegate* delegate, const SdfPath& id,
    const HdMeshTopology& topology, const std::vector<glm::vec3>& positions,
    std::vector<Vertex>& vertices, std::vector<uint32_t>* indices,
    std::vector<Face>* faces, const SdfPath& requestedMaterialId,
    std::vector<SdfPath>* materialSlotPaths,
    const VtValue* prefetchedSt = nullptr)
{
    const MeshPrimvars primvars =
        ReadMeshPrimvars(delegate, id, prefetchedSt);
    const VtIntArray& counts = topology.GetFaceVertexCounts();
    const VtIntArray& sourceIndices = topology.GetFaceVertexIndices();

    std::vector<SdfPath> localMaterialSlotPaths;
    std::vector<SdfPath>& slotPaths =
        materialSlotPaths != nullptr ? *materialSlotPaths : localMaterialSlotPaths;
    const std::vector<int> faceSlots = (faces != nullptr)
        ? BuildFaceMaterialSlots(topology, requestedMaterialId, counts.size(), slotPaths)
        : std::vector<int>();

    // Blender 5.2 emits one right-handed Hydra submesh per material and has
    // already converted every polygon to a corner triangle. This is by far the
    // common path: size the outputs once and fill them directly, without a
    // generic fan-triangulation loop or repeated vector growth.
    const bool directTriangles =
        topology.GetOrientation() == HdTokens->rightHanded
        && topology.GetHoleIndices().empty()
        && sourceIndices.size() == counts.size() * 3
        && std::ranges::all_of(counts, [](const int count) {
               return count == 3;
           })
        && std::ranges::all_of(sourceIndices, [&](const int pointIndex) {
               return pointIndex >= 0
                   && static_cast<size_t>(pointIndex) < positions.size();
           });
    if (directTriangles) {
        const size_t triangleCount = counts.size();
        const size_t cornerCount = sourceIndices.size();
        vertices.resize(cornerCount);
        if (indices != nullptr)
            indices->resize(cornerCount);
        if (faces != nullptr) {
            faces->resize(triangleCount);
            for (size_t faceIndex = 0; faceIndex < triangleCount; ++faceIndex)
                (*faces)[faceIndex] = Face{faceSlots[faceIndex]};
        }

        const auto populateRange = [&](const size_t begin, const size_t end) {
            for (size_t faceIndex = begin; faceIndex < end; ++faceIndex) {
                const size_t base = faceIndex * 3;
                const std::array<int, 3> pointIndices{
                    sourceIndices[base], sourceIndices[base + 1],
                    sourceIndices[base + 2]};
                const std::array<size_t, 3> cornerIndices{
                    base, base + 1, base + 2};
                PopulateTriangle(
                    positions, primvars, pointIndices, cornerIndices, faceIndex,
                    vertices.data() + base);
                if (indices != nullptr) {
                    (*indices)[base] = static_cast<uint32_t>(base);
                    (*indices)[base + 1] =
                        static_cast<uint32_t>(base + 1);
                    (*indices)[base + 2] =
                        static_cast<uint32_t>(base + 2);
                }
            }
        };

        // Hydra already parallelizes Sync across Rprims. Only split the inner
        // loop for a genuinely large submesh, and use a large grain so nested
        // parallelism does not turn scenes with many ordinary meshes into a
        // flood of tiny tasks.
        if (triangleCount >= ParallelTriangleThreshold) {
            tbb::parallel_for(
                tbb::blocked_range<size_t>(
                    0, triangleCount, ParallelTriangleGrainSize),
                [&populateRange](const tbb::blocked_range<size_t>& range) {
                    populateRange(range.begin(), range.end());
                });
        } else {
            populateRange(0, triangleCount);
        }
        return;
    }

    size_t maximumTriangleCount = 0;
    for (const int count : counts) {
        if (count >= 3)
            maximumTriangleCount += static_cast<size_t>(count - 2);
    }
    vertices.reserve(maximumTriangleCount * 3);
    if (indices != nullptr)
        indices->reserve(maximumTriangleCount * 3);
    if (faces != nullptr)
        faces->reserve(maximumTriangleCount);

    const bool flip = topology.GetOrientation() == HdTokens->leftHanded;
    size_t sourceOffset = 0;
    for (size_t faceIndex = 0; faceIndex < counts.size(); ++faceIndex) {
        const int count = counts[faceIndex];
        if (count < 3 || sourceOffset + static_cast<size_t>(count) > sourceIndices.size()) {
            sourceOffset += std::max(count, 0);
            continue;
        }
        for (int fan = 1; fan + 1 < count; ++fan) {
            int localCorners[3] = {0, fan, fan + 1};
            if (flip)
                std::swap(localCorners[1], localCorners[2]);
            int pointIndices[3]{};
            bool valid = true;
            for (int c = 0; c < 3; ++c) {
                pointIndices[c] = sourceIndices[sourceOffset + localCorners[c]];
                valid &= pointIndices[c] >= 0
                    && static_cast<size_t>(pointIndices[c]) < positions.size();
            }
            if (!valid)
                continue;
            Vertex triangle[3]{};
            const std::array<int, 3> trianglePointIndices{
                pointIndices[0], pointIndices[1], pointIndices[2]};
            const std::array<size_t, 3> cornerIndices{
                sourceOffset + static_cast<size_t>(localCorners[0]),
                sourceOffset + static_cast<size_t>(localCorners[1]),
                sourceOffset + static_cast<size_t>(localCorners[2])};
            PopulateTriangle(
                positions, primvars, trianglePointIndices, cornerIndices,
                faceIndex, triangle);
            for (int c = 0; c < 3; ++c) {
                if (indices != nullptr)
                    indices->push_back(static_cast<uint32_t>(vertices.size()));
                vertices.push_back(triangle[c]);
            }
            if (faces != nullptr)
                faces->push_back(Face{faceSlots[faceIndex]});
        }
        sourceOffset += static_cast<size_t>(count);
    }
}
}

// ---------------------------------------------------------------------------
// Decode a gaussian-splat path from a UV-encoded marker mesh.
// The encoding stores "GSPLAT\0" + 4-byte LE path length + UTF-8 path bytes
// in the st primvar, with two bytes per UV channel (u=byte0/255, v=byte1/255).
// ---------------------------------------------------------------------------
static std::string SplatNameFromPath(const std::string& path)
{
    const size_t slash = path.find_last_of("/\\");
    std::string name = slash != std::string::npos ? path.substr(slash + 1) : path;
    const size_t dot = name.find_last_of('.');
    if (dot != std::string::npos)
        name = name.substr(0, dot);
    return name;
}

static std::optional<std::string> DecodeSplatPath(const VtValue& value)
{
    if (!value.IsHolding<VtVec2fArray>())
        return std::nullopt;

    const VtVec2fArray& uvs = value.UncheckedGet<VtVec2fArray>();

    constexpr uint8_t magic[] = {'G', 'S', 'P', 'L', 'A', 'T', 0};
    constexpr size_t headerSize = sizeof(magic) + sizeof(uint32_t);
    const size_t byteCount = uvs.size() * 2;
    if (byteCount < headerSize)
        return std::nullopt;

    const auto byteAt = [&uvs](const size_t byteIndex) {
        const GfVec2f& uv = uvs[byteIndex / 2];
        return static_cast<uint8_t>(std::clamp(
            std::lround(uv[byteIndex % 2] * 255.0f), 0L, 255L));
    };
    for (size_t i = 0; i < sizeof(magic); ++i)
        if (byteAt(i) != magic[i])
            return std::nullopt;

    uint32_t pathLength = 0;
    for (size_t i = 0; i < sizeof(pathLength); ++i) {
        pathLength |=
            static_cast<uint32_t>(byteAt(sizeof(magic) + i)) << (i * 8);
    }

    if (pathLength > byteCount - headerSize)
        return std::nullopt;

    std::string path(pathLength, '\0');
    for (size_t i = 0; i < pathLength; ++i)
        path[i] = static_cast<char>(byteAt(headerSize + i));
    return path;
}

HdNoorRayMesh::HdNoorRayMesh(const SdfPath& id)
    : HdMesh(id)
{
}

HdNoorRayMesh::~HdNoorRayMesh() = default;

HdDirtyBits HdNoorRayMesh::GetInitialDirtyBitsMask() const
{
    return HdChangeTracker::InitRepr | HdChangeTracker::DirtyPoints
        | HdChangeTracker::DirtyTopology | HdChangeTracker::DirtyTransform
        | HdChangeTracker::DirtyVisibility | HdChangeTracker::DirtyPrimvar
        | HdChangeTracker::DirtyNormals
        | HdChangeTracker::DirtyMaterialId | HdChangeTracker::DirtyInstancer;
}

HdDirtyBits HdNoorRayMesh::_PropagateDirtyBits(const HdDirtyBits bits) const
{
    return bits;
}

void HdNoorRayMesh::_InitRepr(const TfToken& reprToken, HdDirtyBits*)
{
    const auto found = std::ranges::find_if(
        _reprs, [&reprToken](const auto& repr) { return repr.first == reprToken; });
    if (found == _reprs.end())
        _reprs.emplace_back(reprToken, HdReprSharedPtr());
}

void HdNoorRayMesh::ReleaseInstances(Scene& scene)
{
    for (const SceneObjectHandle object : objects_)
        scene.removeObject(object);
    objects_.clear();
}

void HdNoorRayMesh::UnbindAllMaterials(HdNoorRayRenderParam& param)
{
    for (size_t slot = 0; slot < boundMaterialIds_.size(); ++slot)
        if (!boundMaterialIds_[slot].IsEmpty())
            param.UnbindMaterial(
                boundMaterialIds_[slot], meshAsset_.handle(), static_cast<uint32_t>(slot));
    boundMaterialIds_.clear();
}

void HdNoorRayMesh::ReleaseAll(HdNoorRayRenderParam& param)
{
    ReleaseInstances(param.session.scene);
    UnbindAllMaterials(param);
    meshAsset_.reset();
    gaussianAsset_.reset();
}

void HdNoorRayMesh::Sync(
    HdSceneDelegate* delegate, HdRenderParam* renderParam,
    HdDirtyBits* dirtyBits, const TfToken&)
{
    const auto syncStart = std::chrono::steady_clock::now();
    auto& param = *static_cast<HdNoorRayRenderParam*>(renderParam);
    // Keep host data-source reads and CPU mesh conversion outside the scene
    // lock. They touch only Hydra-owned input and local output buffers. This
    // also lets them run concurrently if mesh parallel Sync is enabled later;
    // scene registry and renderer mutations remain serialized below.
    std::unique_lock lock(param.mutex);
    Scene& scene = param.session.scene;
    const std::string primName = GetId().GetString();

    // --- Gaussian splat detection via UV-encoded marker mesh ---
    std::optional<VtValue> prefetchedSt;
    if (*dirtyBits & HdChangeTracker::DirtyPrimvar) {
        prefetchedSt.emplace(delegate->Get(GetId(), StToken));
        const auto decodedPath = DecodeSplatPath(*prefetchedSt);
        if (decodedPath) {
            if (*decodedPath != splatPath_) {
                // Path changed — drop the old instances and asset so the new
                // file is loaded below and the old splat data is freed.
                ReleaseInstances(scene);
                gaussianAsset_.reset();
                splatPath_ = *decodedPath;
            }
        } else if (!splatPath_.empty()) {
            ReleaseInstances(scene);
            gaussianAsset_.reset();
            splatPath_.clear();
        }
    }

    const bool isGaussian = !splatPath_.empty();

    if (isGaussian) {
        if (meshAsset_.isValid()) {
            UnbindAllMaterials(param);
            ReleaseInstances(scene);
            meshAsset_.reset();
        }

        if (!gaussianAsset_.isValid()) {
            try {
                gaussianAsset_ = scene.add(
                    GaussianAsset::CreateFromFile(scene, primName, splatPath_));
            } catch (const std::exception& error) {
                // A missing or malformed splat file is a scene authoring
                // mistake, not a reason to take the host process down.
                TF_WARN("hdNoorRay could not load Gaussian splat '%s': %s",
                    splatPath_.c_str(), error.what());
                splatPath_.clear();
                UpdateRenderTag(delegate, renderParam);
                *dirtyBits = HdChangeTracker::Clean;
                return;
            }
        }

        const bool instancesDirty = objects_.empty()
            || (*dirtyBits & (HdChangeTracker::DirtyTransform
                | HdChangeTracker::DirtyInstancer)) != 0;
        if (instancesDirty) {
            _UpdateInstancer(delegate, dirtyBits);
            VtMatrix4dArray transforms;
            if (GetInstancerId().IsEmpty()) {
                transforms.push_back(delegate->GetTransform(GetId()));
            } else {
                HdInstancer::_SyncInstancerAndParents(
                    delegate->GetRenderIndex(), GetInstancerId());
                auto* instancer = dynamic_cast<HdNoorRayInstancer*>(
                    delegate->GetRenderIndex().GetInstancer(GetInstancerId()));
                if (instancer != nullptr)
                    transforms = instancer->ComputeInstanceTransforms(GetId());
                const GfMatrix4d prototypeTransform = delegate->GetTransform(GetId());
                for (GfMatrix4d& transform : transforms)
                    transform = prototypeTransform * transform;
                if (transforms.empty())
                    transforms.push_back(prototypeTransform);
            }

            // Resize gaussian instance list to match the number of transforms.
            while (objects_.size() > transforms.size()) {
                scene.removeObject(objects_.back());
                objects_.pop_back();
            }
            while (objects_.size() < transforms.size()) {
                auto instance = std::make_unique<GaussianInstance>(
                    scene, SplatNameFromPath(splatPath_), gaussianAsset_,
                    Transform{});
                objects_.push_back(scene.add(std::move(instance)));
            }
            for (size_t i = 0; i < transforms.size(); ++i)
                if (SceneObject* object = scene.getObject(objects_[i]))
                    object->setWorldTransformFromMatrix(ToGlm(transforms[i]));
        }

        if (*dirtyBits & HdChangeTracker::DirtyVisibility) {
            const bool visible = delegate->GetVisible(GetId());
            for (const SceneObjectHandle object : objects_)
                if (SceneObject* resolved = scene.getObject(object))
                    resolved->setVisible(visible);
        }

        if (!objects_.empty()) {
            scene.setDirtyFlag(TLAS);
            scene.setDirtyFlag(Accumulation);
        }

        UpdateRenderTag(delegate, renderParam);
        *dirtyBits = HdChangeTracker::Clean;
        return;
    }

    // --- Regular mesh code ---
    _UpdateInstancer(delegate, dirtyBits);
    const SdfPath requestedMaterialId = delegate->GetMaterialId(GetId());
    SetMaterialId(requestedMaterialId);

    const bool pointsDirty = (*dirtyBits & HdChangeTracker::DirtyPoints) != 0;
    const bool topologyDirty = (*dirtyBits & HdChangeTracker::DirtyTopology) != 0;
    const bool primvarsDirty = (*dirtyBits & HdChangeTracker::DirtyPrimvar) != 0;
    const bool normalsDirty = (*dirtyBits & HdChangeTracker::DirtyNormals) != 0;
    const bool geometryDirty =
        pointsDirty || topologyDirty || primvarsDirty || normalsDirty;

    // Slot 0 always tracks the Rprim's own material; slots 1+ (from
    // HdGeomSubsets) are only re-derived when topology is actually rebuilt
    // below, since HdGeomSubsets live inside HdMeshTopology itself and are
    // otherwise not re-read this Sync (same caveat single-material meshes
    // already had for anything embedded in topology).
    std::vector<SdfPath> desiredMaterialIds = boundMaterialIds_;
    if (desiredMaterialIds.empty())
        desiredMaterialIds.push_back(requestedMaterialId);
    else
        desiredMaterialIds[0] = requestedMaterialId;

    if (geometryDirty) {
        MeshAsset* asset = meshAsset_.get();
        const bool rebuildTopology = asset == nullptr || topologyDirty;
        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices;
        std::vector<Face> faces;
        std::vector<SdfPath> materialSlotPaths;
        lock.unlock();
        const std::vector<glm::vec3> positions =
            Vec3Values(delegate->Get(GetId(), HdTokens->points));
        BuildTriangleMesh(
            delegate, GetId(), GetMeshTopology(delegate), positions, vertices,
            rebuildTopology ? &indices : nullptr,
            rebuildTopology ? &faces : nullptr,
            requestedMaterialId, rebuildTopology ? &materialSlotPaths : nullptr,
            prefetchedSt ? &*prefetchedSt : nullptr);
        lock.lock();

        if (rebuildTopology)
            desiredMaterialIds = std::move(materialSlotPaths);

        // The scene's managed registry can move while the lock is released.
        asset = meshAsset_.get();
        if (asset != nullptr && !topologyDirty) {
            // Positions affect geometric normals and the tangent frame, so a
            // point edit must refresh complete vertex data. Updating positions
            // alone leaves normal-mapped and flat-shaded deformations stale.
            if (!vertices.empty())
                asset->updateVertexData(vertices);
        } else if (!vertices.empty() && !indices.empty()) {
            // Topology changed or first sync — build new index/face data too.
            // BindMaterial (below, once each slot's material Sprim has synced
            // and published) is what points this mesh at its real materials.
            // Until that happens, a shared native grey material slot (owned
            // and compiled once by the render param) avoids a hole.
            const MaterialRef material = param.GetNativeGreyMaterial();
            if (asset) {
                asset->replaceGeometry(vertices, indices, faces,
                    static_cast<uint32_t>(desiredMaterialIds.size()));
            } else {
                meshAsset_ = scene.add(MeshAsset(
                    scene, GetId().GetString(), vertices, indices, faces,
                    std::vector<MaterialRef>(
                        std::max<size_t>(desiredMaterialIds.size(), 1), material)));
            }
        }
    }

    if (meshAsset_.isValid() && boundMaterialIds_ != desiredMaterialIds) {
        const size_t slotCount =
            std::max(boundMaterialIds_.size(), desiredMaterialIds.size());
        for (size_t slot = 0; slot < slotCount; ++slot) {
            const SdfPath oldId =
                slot < boundMaterialIds_.size() ? boundMaterialIds_[slot] : SdfPath();
            const SdfPath newId =
                slot < desiredMaterialIds.size() ? desiredMaterialIds[slot] : SdfPath();
            if (oldId == newId)
                continue;
            if (!oldId.IsEmpty())
                param.UnbindMaterial(oldId, meshAsset_.handle(), static_cast<uint32_t>(slot));
            if (!newId.IsEmpty())
                param.BindMaterial(newId, meshAsset_.handle(), static_cast<uint32_t>(slot));
        }
        boundMaterialIds_ = desiredMaterialIds;
    }

    const bool instancesDirty = objects_.empty()
        || (*dirtyBits & (HdChangeTracker::DirtyTransform
            | HdChangeTracker::DirtyInstancer)) != 0;
    if (meshAsset_.isValid() && instancesDirty) {
        VtMatrix4dArray transforms;
        if (GetInstancerId().IsEmpty()) {
            transforms.push_back(delegate->GetTransform(GetId()));
        } else {
            HdInstancer::_SyncInstancerAndParents(
                delegate->GetRenderIndex(), GetInstancerId());
            auto* instancer = dynamic_cast<HdNoorRayInstancer*>(
                delegate->GetRenderIndex().GetInstancer(GetInstancerId()));
            if (instancer != nullptr)
                transforms = instancer->ComputeInstanceTransforms(GetId());
            const GfMatrix4d prototypeTransform = delegate->GetTransform(GetId());
            for (GfMatrix4d& transform : transforms)
                transform = prototypeTransform * transform;
        }
        while (objects_.size() > transforms.size()) {
            scene.removeObject(objects_.back());
            objects_.pop_back();
        }
        while (objects_.size() < transforms.size()) {
            auto instance = std::make_unique<MeshInstance>(
                scene, GetId().GetString(), meshAsset_, Transform());
            objects_.push_back(scene.add(std::move(instance)));
        }
        for (size_t i = 0; i < transforms.size(); ++i)
            if (SceneObject* object = scene.getObject(objects_[i]))
                object->setWorldTransformFromMatrix(ToGlm(transforms[i]));
    }

    if (*dirtyBits & HdChangeTracker::DirtyVisibility) {
        const bool visible = delegate->GetVisible(GetId());
        for (const SceneObjectHandle object : objects_)
            if (SceneObject* resolved = scene.getObject(object))
                resolved->setVisible(visible);
        scene.setDirtyFlag(TLAS);
        scene.setDirtyFlag(Accumulation);
    }
    UpdateRenderTag(delegate, renderParam);
    *dirtyBits = HdChangeTracker::Clean;
    if (std::getenv("NR_PROFILE_SYNC")) {
        const double milliseconds =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - syncStart).count();
        fprintf(stderr, "NR_PROFILE mesh %.3f %s\n",
            milliseconds, GetId().GetText());
    }
}

void HdNoorRayMesh::Finalize(HdRenderParam* renderParam)
{
    auto& param = *static_cast<HdNoorRayRenderParam*>(renderParam);
    std::scoped_lock lock(param.mutex);
    ReleaseAll(param);
    splatPath_.clear();
}

PXR_NAMESPACE_CLOSE_SCOPE
