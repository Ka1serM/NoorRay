#include "mesh.h"

#include "instancer.h"
#include "renderParam.h"

#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3d.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/imaging/hd/changeTracker.h>
#include <pxr/imaging/hd/renderIndex.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/imaging/hd/tokens.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <optional>
#include <vector>

#include <glm/geometric.hpp>

#include "Mesh/Assets/GaussianAsset.h"
#include "Mesh/Assets/MeshAsset.h"
#include "Mesh/Transform.h"
#include "Scene/GaussianInstance.h"
#include "Scene/MeshInstance.h"
#include "Scene/Scene.h"

#include <pxr/base/tf/diagnostic.h>
#include <exception>

PXR_NAMESPACE_OPEN_SCOPE

namespace
{
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
        for (const GfVec3f& v : value.UncheckedGet<VtVec3fArray>())
            result.emplace_back(v[0], v[1], v[2]);
    } else if (value.IsHolding<VtVec3dArray>()) {
        for (const GfVec3d& v : value.UncheckedGet<VtVec3dArray>())
            result.emplace_back(v[0], v[1], v[2]);
    }
    return result;
}

std::vector<glm::vec2> Vec2Values(const VtValue& value)
{
    std::vector<glm::vec2> result;
    if (value.IsHolding<VtVec2fArray>()) {
        for (const GfVec2f& v : value.UncheckedGet<VtVec2fArray>())
            result.emplace_back(v[0], v[1]);
    }
    return result;
}

HdInterpolation FindInterpolation(
    HdSceneDelegate* delegate, const SdfPath& id, const TfToken& name)
{
    for (const HdInterpolation interpolation : {
             HdInterpolationFaceVarying, HdInterpolationVertex,
             HdInterpolationVarying, HdInterpolationUniform,
             HdInterpolationConstant}) {
        for (const HdPrimvarDescriptor& descriptor :
             delegate->GetPrimvarDescriptors(id, interpolation))
            if (descriptor.name == name)
                return interpolation;
    }
    return HdInterpolationConstant;
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

void BuildTriangleMesh(
    HdSceneDelegate* delegate, const SdfPath& id,
    const HdMeshTopology& topology, const std::vector<glm::vec3>& positions,
    std::vector<Vertex>& vertices, std::vector<uint32_t>& indices,
    std::vector<Face>& faces,
    std::vector<uint32_t>* vertexToPointIndex = nullptr)
{
    const std::vector<glm::vec3> normals =
        Vec3Values(delegate->Get(id, HdTokens->normals));
    const std::vector<glm::vec2> uvs =
        Vec2Values(delegate->Get(id, TfToken("st")));
    const HdInterpolation normalInterpolation =
        FindInterpolation(delegate, id, HdTokens->normals);
    const HdInterpolation uvInterpolation =
        FindInterpolation(delegate, id, TfToken("st"));

    const VtIntArray& counts = topology.GetFaceVertexCounts();
    const VtIntArray& sourceIndices = topology.GetFaceVertexIndices();
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
            const glm::vec3 geometricNormal = glm::normalize(glm::cross(
                positions[pointIndices[1]] - positions[pointIndices[0]],
                positions[pointIndices[2]] - positions[pointIndices[0]]));
            Vertex triangle[3]{};
            for (int c = 0; c < 3; ++c) {
                const size_t cornerIndex = sourceOffset + localCorners[c];
                triangle[c].position = positions[pointIndices[c]];
                triangle[c].normal = PrimvarAt(
                    normals, normalInterpolation, pointIndices[c], cornerIndex,
                    faceIndex, geometricNormal);
                if (glm::length(triangle[c].normal) > 0.0f)
                    triangle[c].normal = glm::normalize(triangle[c].normal);
                triangle[c].uv = PrimvarAt(
                    uvs, uvInterpolation, pointIndices[c], cornerIndex,
                    faceIndex, glm::vec2(0.0f));
                triangle[c].tangentSign = 1.0f;
            }
            const glm::vec3 edge1 = triangle[1].position - triangle[0].position;
            const glm::vec3 edge2 = triangle[2].position - triangle[0].position;
            const glm::vec2 duv1 = triangle[1].uv - triangle[0].uv;
            const glm::vec2 duv2 = triangle[2].uv - triangle[0].uv;
            const float determinant = duv1.x * duv2.y - duv1.y * duv2.x;
            const glm::vec3 tangent = std::abs(determinant) > 1e-8f
                ? glm::normalize((edge1 * duv2.y - edge2 * duv1.y) / determinant)
                : glm::normalize(edge1);
            for (int c = 0; c < 3; ++c) {
                Vertex& vertex = triangle[c];
                vertex.tangent = tangent;
                indices.push_back(static_cast<uint32_t>(vertices.size()));
                vertices.push_back(vertex);
                if (vertexToPointIndex)
                    vertexToPointIndex->push_back(
                        static_cast<uint32_t>(pointIndices[c]));
            }
            faces.push_back({0});
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

    std::vector<uint8_t> bytes;
    bytes.reserve(uvs.size() * 2);
    for (const GfVec2f& uv : uvs) {
        bytes.push_back(static_cast<uint8_t>(
            std::clamp(std::lround(uv[0] * 255.0f), 0L, 255L)));
        bytes.push_back(static_cast<uint8_t>(
            std::clamp(std::lround(uv[1] * 255.0f), 0L, 255L)));
    }

    constexpr uint8_t magic[] = {'G', 'S', 'P', 'L', 'A', 'T', 0};
    constexpr size_t headerSize = sizeof(magic) + sizeof(uint32_t);

    if (bytes.size() < headerSize
        || std::memcmp(bytes.data(), magic, sizeof(magic)) != 0)
        return std::nullopt;

    uint32_t pathLength = 0;
    std::memcpy(&pathLength, bytes.data() + sizeof(magic), sizeof(pathLength));

    if (pathLength > bytes.size() - headerSize)
        return std::nullopt;

    return std::string(
        reinterpret_cast<const char*>(bytes.data() + headerSize), pathLength);
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

void HdNoorRayMesh::ReleaseAll(HdNoorRayRenderParam& param)
{
    ReleaseInstances(param.session.scene);
    param.UnbindMaterial(boundMaterialId_, meshAsset_.handle());
    boundMaterialId_ = SdfPath();
    // Dropping the asset references is what actually frees the geometry, the
    // acceleration structure and the splat data.
    meshAsset_.reset();
    gaussianAsset_.reset();
}

void HdNoorRayMesh::Sync(
    HdSceneDelegate* delegate, HdRenderParam* renderParam,
    HdDirtyBits* dirtyBits, const TfToken&)
{
    auto& param = *static_cast<HdNoorRayRenderParam*>(renderParam);
    std::scoped_lock lock(param.mutex);
    Scene& scene = param.session.scene;
    const std::string primName = GetId().GetString();

    // --- Gaussian splat detection via UV-encoded marker mesh ---
    if (*dirtyBits & HdChangeTracker::DirtyPrimvar) {
        const VtValue st = delegate->Get(GetId(), TfToken("st"));
        const auto decodedPath = DecodeSplatPath(st);
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
            param.UnbindMaterial(boundMaterialId_, meshAsset_.handle());
            boundMaterialId_ = SdfPath();
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
                param.MarkSceneDirty();
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
        param.MarkSceneDirty();
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
    const bool geometryDirty = pointsDirty || topologyDirty || primvarsDirty;

    if (geometryDirty) {
        MeshAsset* asset = meshAsset_.get();

        // Position-only fast path: reuse cached topology mapping to update
        // expanded vertex positions without re-running BuildTriangleMesh.
        if (asset && pointsDirty && !topologyDirty && !primvarsDirty
            && !vertexToPointIndex_.empty())
        {
            const std::vector<glm::vec3> positions =
                Vec3Values(delegate->Get(GetId(), HdTokens->points));
            std::vector<glm::vec3> newExpandedPositions(vertexToPointIndex_.size());
            for (size_t i = 0; i < vertexToPointIndex_.size(); ++i)
            {
                const uint32_t pointIdx = vertexToPointIndex_[i];
                newExpandedPositions[i] = pointIdx < positions.size()
                    ? positions[pointIdx] : glm::vec3(0.0f);
            }
            asset->updatePositions(newExpandedPositions);
        }
        else if (asset && geometryDirty && !topologyDirty)
        {
            // Vertex data changed but topology is the same — rebuild vertex
            // array and use updateVertexData for a BLAS refit instead of
            // a full rebuild.
            const std::vector<glm::vec3> positions =
                Vec3Values(delegate->Get(GetId(), HdTokens->points));
            std::vector<Vertex> vertices;
            std::vector<uint32_t> indices;
            std::vector<Face> faces;
            vertexToPointIndex_.clear();
            BuildTriangleMesh(
                delegate, GetId(), GetMeshTopology(delegate), positions,
                vertices, indices, faces, &vertexToPointIndex_);
            if (!vertices.empty() && !indices.empty())
                asset->updateVertexData(vertices);
        }
        else
        {
            // Topology changed or first sync — full rebuild.
            const std::vector<glm::vec3> positions =
                Vec3Values(delegate->Get(GetId(), HdTokens->points));
            std::vector<Vertex> vertices;
            std::vector<uint32_t> indices;
            std::vector<Face> faces;
            vertexToPointIndex_.clear();
            BuildTriangleMesh(
                delegate, GetId(), GetMeshTopology(delegate), positions,
                vertices, indices, faces, &vertexToPointIndex_);
            if (!vertices.empty() && !indices.empty()) {
                Material material;
                material.albedo = glm::vec3(0.8f);
                material.roughness = 0.5f;
                if (asset) {
                    asset->replaceGeometry(vertices, indices, faces);
                } else {
                    meshAsset_ = scene.add(MeshAsset(
                        scene, GetId().GetString(), vertices, indices, faces,
                        {material}));
                }
            }
        }
    }

    if (meshAsset_.isValid() && boundMaterialId_ != requestedMaterialId) {
        param.UnbindMaterial(boundMaterialId_, meshAsset_.handle());
        boundMaterialId_ = requestedMaterialId;
        param.BindMaterial(boundMaterialId_, meshAsset_.handle());
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
    param.MarkSceneDirty();
    *dirtyBits = HdChangeTracker::Clean;
}

void HdNoorRayMesh::Finalize(HdRenderParam* renderParam)
{
    auto& param = *static_cast<HdNoorRayRenderParam*>(renderParam);
    std::scoped_lock lock(param.mutex);
    ReleaseAll(param);
    splatPath_.clear();
    param.MarkSceneDirty();
}

PXR_NAMESPACE_CLOSE_SCOPE
