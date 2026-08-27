#pragma once

#include "api.h"

#include <pxr/base/gf/vec3i.h>
#include <pxr/imaging/hd/renderBuffer.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

PXR_NAMESPACE_OPEN_SCOPE

class HDNOORRAY_API HdNoorRayRenderBuffer final : public HdRenderBuffer
{
public:
    explicit HdNoorRayRenderBuffer(const SdfPath& id);
    ~HdNoorRayRenderBuffer() override;

    bool Allocate(
        const GfVec3i& dimensions, HdFormat format, bool multiSampled) override;
    unsigned int GetWidth() const override { return width_; }
    unsigned int GetHeight() const override { return height_; }
    unsigned int GetDepth() const override { return 1; }
    HdFormat GetFormat() const override { return format_; }
    bool IsMultiSampled() const override { return multiSampled_; }

    void* Map() override;
    void Unmap() override;
    bool IsMapped() const override;
    void Resolve() override {}
    bool IsConverged() const override;

    void SetConverged(bool converged);
    void WriteFloatPixel(size_t pixel, const float* values, size_t componentCount);
    void ClearFloat(const float* values, size_t componentCount);
    // Copy an already synchronized CPU image into this Hydra AOV buffer.
    // libnoorray owns all GPU transfers; the delegate never exposes backend
    // resources.
    bool CopyFromHost(const void* source, size_t bytes);

protected:
    void _Deallocate() override;

private:
    unsigned int width_{};
    unsigned int height_{};
    HdFormat format_{HdFormatInvalid};
    bool multiSampled_{};
    std::vector<std::byte> data_;
    std::atomic<int> mapCount_{};
    std::atomic<bool> converged_{};
};

PXR_NAMESPACE_CLOSE_SCOPE
