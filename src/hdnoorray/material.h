#pragma once

#include "api.h"

#include <pxr/imaging/hd/material.h>

#include <cstdint>
#include <memory>

namespace MaterialX_v1_39_4
{
class Document;
using DocumentPtr = std::shared_ptr<Document>;
}
namespace MaterialX = MaterialX_v1_39_4;

PXR_NAMESPACE_OPEN_SCOPE

// Shared native grey fallback document (open_pbr_surface, albedo 0.8,
// roughness 0.5), used by every path that needs a visible material where none
// is authored. Static storage is immutable after first construction; callers
// must never mutate it (setDataLibrary from a compile is idempotent).
const MaterialX::DocumentPtr& GetSharedNativeFallbackMaterial();

class HDNOORRAY_API HdNoorRayMaterial final : public HdMaterial
{
public:
    explicit HdNoorRayMaterial(const SdfPath& id);

    void Sync(HdSceneDelegate*, HdRenderParam*, HdDirtyBits*) override;
    void Finalize(HdRenderParam*) override;
    HdDirtyBits GetInitialDirtyBitsMask() const override;

private:
    // A material Sprim can be dirtied for reasons unrelated to its custom
    // XML transport setting. Remember the last queued immutable snapshot so
    // an identical document never gets parsed or compiled twice.
    bool usingCustomDocument_{};
    uint64_t customDocumentHash_{};
    uint64_t customDocumentRevision_{};
};

PXR_NAMESPACE_CLOSE_SCOPE
