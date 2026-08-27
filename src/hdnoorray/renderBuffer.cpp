#include "renderBuffer.h"

#include <pxr/base/gf/half.h>
#include <pxr/imaging/hd/types.h>

#include <algorithm>
#include <cstring>

PXR_NAMESPACE_OPEN_SCOPE

HdNoorRayRenderBuffer::HdNoorRayRenderBuffer(const SdfPath& id)
    : HdRenderBuffer(id)
{
}

HdNoorRayRenderBuffer::~HdNoorRayRenderBuffer() = default;

bool HdNoorRayRenderBuffer::Allocate(
    const GfVec3i& dimensions,
    const HdFormat format,
    const bool multiSampled)
{
    _Deallocate();
    if (dimensions[0] <= 0 || dimensions[1] <= 0 || dimensions[2] != 1
        || format == HdFormatInvalid)
        return false;

    width_ = static_cast<unsigned int>(dimensions[0]);
    height_ = static_cast<unsigned int>(dimensions[1]);
    format_ = format;
    multiSampled_ = multiSampled;
    data_.resize(static_cast<size_t>(width_) * height_
        * HdDataSizeOfFormat(format_));
    return true;
}

void* HdNoorRayRenderBuffer::Map()
{
    mapCount_.fetch_add(1, std::memory_order_relaxed);
    return data_.data();
}

void HdNoorRayRenderBuffer::Unmap()
{
    mapCount_.fetch_sub(1, std::memory_order_relaxed);
}

bool HdNoorRayRenderBuffer::IsMapped() const
{
    return mapCount_.load(std::memory_order_relaxed) != 0;
}

bool HdNoorRayRenderBuffer::IsConverged() const
{
    return converged_.load(std::memory_order_acquire);
}

void HdNoorRayRenderBuffer::SetConverged(const bool converged)
{
    converged_.store(converged, std::memory_order_release);
}

void HdNoorRayRenderBuffer::WriteFloatPixel(
    const size_t pixel,
    const float* values,
    const size_t valueComponentCount)
{
    if (pixel >= static_cast<size_t>(width_) * height_)
        return;

    const size_t componentCount = HdGetComponentCount(format_);
    std::byte* destination = data_.data()
        + pixel * HdDataSizeOfFormat(format_);
    const HdFormat componentFormat = HdGetComponentFormat(format_);
    for (size_t component = 0; component < componentCount; ++component) {
        const float value = component < valueComponentCount
            ? values[component] : 0.0f;
        if (componentFormat == HdFormatFloat32)
            reinterpret_cast<float*>(destination)[component] = value;
        else if (componentFormat == HdFormatFloat16)
            reinterpret_cast<uint16_t*>(destination)[component] =
                GfHalf(value).bits();
    }
}

void HdNoorRayRenderBuffer::ClearFloat(
    const float* values, const size_t componentCount)
{
    const size_t pixelCount = static_cast<size_t>(width_) * height_;
    for (size_t pixel = 0; pixel < pixelCount; ++pixel)
        WriteFloatPixel(pixel, values, componentCount);
}

bool HdNoorRayRenderBuffer::CopyFromHost(
    const void* source, const size_t bytes)
{
    const size_t expectedBytes = static_cast<size_t>(width_) * height_
        * sizeof(float) * 4;
    if (source == nullptr || bytes < expectedBytes
        || format_ != HdFormatFloat32Vec4 || data_.empty())
        return false;
    std::memcpy(data_.data(), source, expectedBytes);
    return true;
}

void HdNoorRayRenderBuffer::_Deallocate()
{
    width_ = 0;
    height_ = 0;
    format_ = HdFormatInvalid;
    multiSampled_ = false;
    data_.clear();
    mapCount_.store(0, std::memory_order_relaxed);
    converged_.store(false, std::memory_order_relaxed);
}

PXR_NAMESPACE_CLOSE_SCOPE
