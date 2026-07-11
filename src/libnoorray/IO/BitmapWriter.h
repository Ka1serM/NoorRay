#pragma once

#include <string>

class Bitmap;

enum class BitmapFormat
{
    Automatic,
    OpenExr,
    RadianceHdr,
    Png,
};

struct BitmapWriteOptions
{
    BitmapFormat format{BitmapFormat::Automatic};
    bool exrHalfFloat{false};
    float pngExposure{};
};

class BitmapWriter
{
public:
    static bool write(
        const std::string& path,
        const Bitmap& bitmap,
        const BitmapWriteOptions& options = {},
        std::string* errorMessage = nullptr);
};
