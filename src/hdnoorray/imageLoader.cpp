#include "imageLoader.h"

#include <pxr/base/gf/half.h>
#include <pxr/imaging/hio/image.h>
#include <pxr/imaging/hio/types.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

PXR_NAMESPACE_OPEN_SCOPE

namespace
{
bool Fail(HdNoorRayDecodedImage* image, std::string* error,
    std::string message)
{
    if (image)
        *image = {};
    if (error)
        *error = std::move(message);
    return false;
}

bool CheckedProduct(
    const size_t lhs, const size_t rhs, size_t* product) noexcept
{
    if (rhs != 0 && lhs > std::numeric_limits<size_t>::max() / rhs)
        return false;
    *product = lhs * rhs;
    return true;
}

template<typename T>
uint8_t NormalizeInteger(const T value)
{
    static_assert(std::is_integral_v<T>);
    if constexpr (std::is_same_v<T, uint8_t>) {
        return value;
    } else {
        constexpr long double minimum =
            static_cast<long double>(std::numeric_limits<T>::lowest());
        constexpr long double maximum =
            static_cast<long double>(std::numeric_limits<T>::max());
        const long double normalized =
            (static_cast<long double>(value) - minimum) / (maximum - minimum);
        return static_cast<uint8_t>(
            std::clamp(std::lround(normalized * 255.0L), 0L, 255L));
    }
}

template<typename Source, typename Convert>
void ExpandToRgba(const Source* source, const size_t pixelCount,
    const int channelCount, const typename std::invoke_result_t<Convert, Source>
        opaqueValue,
    std::vector<typename std::invoke_result_t<Convert, Source>>& destination,
    Convert convert)
{
    using Destination = typename std::invoke_result_t<Convert, Source>;
    destination.resize(pixelCount * 4);
    for (size_t pixel = 0; pixel < pixelCount; ++pixel) {
        const Source* input = source + pixel * channelCount;
        Destination* output = destination.data() + pixel * 4;
        if (channelCount == 1) {
            output[0] = output[1] = output[2] = convert(input[0]);
            output[3] = opaqueValue;
        } else if (channelCount == 2) {
            output[0] = output[1] = output[2] = convert(input[0]);
            output[3] = convert(input[1]);
        } else {
            output[0] = convert(input[0]);
            output[1] = convert(input[1]);
            output[2] = convert(input[2]);
            output[3] =
                channelCount == 4 ? convert(input[3]) : opaqueValue;
        }
    }
}

template<typename Source>
void ExpandInteger(const std::vector<std::byte>& source,
    const size_t pixelCount, const int channelCount,
    HdNoorRayDecodedImage* image)
{
    image->pixelType = HdNoorRayImagePixelType::Rgba8;
    ExpandToRgba(reinterpret_cast<const Source*>(source.data()), pixelCount,
        channelCount, uint8_t{255}, image->rgba8,
        [](const Source value) { return NormalizeInteger(value); });
}

template<typename Source>
void ExpandFloat(const std::vector<std::byte>& source,
    const size_t pixelCount, const int channelCount,
    HdNoorRayDecodedImage* image)
{
    image->pixelType = HdNoorRayImagePixelType::Rgba32Float;
    ExpandToRgba(reinterpret_cast<const Source*>(source.data()), pixelCount,
        channelCount, 1.0f, image->rgba32Float,
        [](const Source value) { return static_cast<float>(value); });
}

void ExpandHalf(const std::vector<std::byte>& source,
    const size_t pixelCount, const int channelCount,
    HdNoorRayDecodedImage* image)
{
    image->pixelType = HdNoorRayImagePixelType::Rgba16Float;
    const uint16_t opaque = GfHalf(1.0f).bits();
    ExpandToRgba(reinterpret_cast<const GfHalf*>(source.data()), pixelCount,
        channelCount, opaque, image->rgba16Float,
        [](const GfHalf value) {
            return static_cast<uint16_t>(value.bits());
        });
}
}

bool HdNoorRayLoadImage(const std::string& assetPath,
    HdNoorRayDecodedImage* image, std::string* error)
{
    if (!image)
        return Fail(nullptr, error, "Image output pointer is null");
    *image = {};
    if (error)
        error->clear();
    if (assetPath.empty())
        return Fail(image, error, "Image asset path is empty");

    try {
        const HioImageSharedPtr source = HioImage::OpenForReading(assetPath,
            0, 0, HioImage::SourceColorSpace::Raw,
            /*suppressErrors=*/true);
        if (!source)
            return Fail(image, error,
                "Hio could not open image asset '" + assetPath + "'");

        const int width = source->GetWidth();
        const int height = source->GetHeight();
        if (width <= 0 || height <= 0)
            return Fail(image, error, "Image asset '" + assetPath
                    + "' has invalid dimensions");

        const HioFormat format = source->GetFormat();
        if (format == HioFormatInvalid || HioIsCompressed(format))
            return Fail(image, error, "Image asset '" + assetPath
                    + "' has an unsupported Hio storage format");

        const int channelCount = HioGetComponentCount(format);
        if (channelCount < 1 || channelCount > 4)
            return Fail(image, error, "Image asset '" + assetPath
                    + "' does not have between one and four channels");

        const HioType componentType = HioGetHioType(format);
        const size_t bytesPerPixel = HioGetDataSizeOfFormat(format);
        const size_t bytesPerComponent = HioGetDataSizeOfType(componentType);
        if (bytesPerPixel == 0 || bytesPerComponent == 0
            || bytesPerPixel
                != bytesPerComponent * static_cast<size_t>(channelCount)) {
            return Fail(image, error, "Image asset '" + assetPath
                    + "' has an unsupported Hio pixel layout");
        }

        size_t pixelCount{};
        size_t nativeByteCount{};
        if (!CheckedProduct(static_cast<size_t>(width),
                static_cast<size_t>(height), &pixelCount)
            || !CheckedProduct(pixelCount, bytesPerPixel, &nativeByteCount)) {
            return Fail(image, error,
                "Image asset '" + assetPath + "' is too large to decode");
        }

        // Four-channel formats can be decoded directly into their final
        // caller-owned vector. Only layouts that need channel expansion or
        // type conversion use the native staging allocation.
        const bool directByte = channelCount == 4
            && (componentType == HioTypeUnsignedByte
                || componentType == HioTypeUnsignedByteSRGB);
        const bool directHalf =
            channelCount == 4 && componentType == HioTypeHalfFloat;
        const bool directFloat =
            channelCount == 4 && componentType == HioTypeFloat;

        std::vector<std::byte> nativePixels;
        void* destination = nullptr;
        if (directByte) {
            image->pixelType = HdNoorRayImagePixelType::Rgba8;
            image->rgba8.resize(pixelCount * 4);
            destination = image->rgba8.data();
        } else if (directHalf) {
            image->pixelType = HdNoorRayImagePixelType::Rgba16Float;
            image->rgba16Float.resize(pixelCount * 4);
            destination = image->rgba16Float.data();
        } else if (directFloat) {
            image->pixelType = HdNoorRayImagePixelType::Rgba32Float;
            image->rgba32Float.resize(pixelCount * 4);
            destination = image->rgba32Float.data();
        } else {
            nativePixels.resize(nativeByteCount);
            destination = nativePixels.data();
        }

        HioImage::StorageSpec storage;
        storage.width = width;
        storage.height = height;
        storage.depth = 1;
        storage.format = format;
        storage.flipped = false;
        storage.data = destination;
        if (!source->Read(storage))
            return Fail(image, error,
                "Hio could not read image asset '" + assetPath + "'");

        image->width = width;
        image->height = height;
        if (directByte || directHalf || directFloat)
            return true;

        switch (componentType) {
        case HioTypeUnsignedByte:
        case HioTypeUnsignedByteSRGB:
            ExpandInteger<uint8_t>(
                nativePixels, pixelCount, channelCount, image);
            break;
        case HioTypeSignedByte:
            ExpandInteger<int8_t>(
                nativePixels, pixelCount, channelCount, image);
            break;
        case HioTypeUnsignedShort:
            ExpandInteger<uint16_t>(
                nativePixels, pixelCount, channelCount, image);
            break;
        case HioTypeSignedShort:
            ExpandInteger<int16_t>(
                nativePixels, pixelCount, channelCount, image);
            break;
        case HioTypeUnsignedInt:
            ExpandInteger<uint32_t>(
                nativePixels, pixelCount, channelCount, image);
            break;
        case HioTypeInt:
            ExpandInteger<int32_t>(
                nativePixels, pixelCount, channelCount, image);
            break;
        case HioTypeHalfFloat:
            ExpandHalf(nativePixels, pixelCount, channelCount, image);
            break;
        case HioTypeFloat:
            ExpandFloat<float>(
                nativePixels, pixelCount, channelCount, image);
            break;
        case HioTypeDouble:
            ExpandFloat<double>(
                nativePixels, pixelCount, channelCount, image);
            break;
        default:
            return Fail(image, error, "Image asset '" + assetPath
                    + "' has an unsupported Hio component type");
        }
        return true;
    } catch (const std::exception& exception) {
        return Fail(image, error, "Could not decode image asset '" + assetPath
                + "': " + exception.what());
    } catch (...) {
        return Fail(image, error,
            "Could not decode image asset '" + assetPath + "'");
    }
}

PXR_NAMESPACE_CLOSE_SCOPE
