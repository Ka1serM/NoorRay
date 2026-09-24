#pragma once

#include <string>

#include "Texture/Texture.h"

// Loads an image file into a Texture: OpenEXR and Radiance HDR as float RGBA,
// every other stb_image format as RGBA8 in the requested encoding.
class TextureReader
{
public:
    static Texture read(const std::string& path,
        TextureEncoding encoding = TextureEncoding::Linear8);
};
