#pragma once

#include <string>

#include "IO/Bitmap.h"

class BitmapReader
{
public:
    static Bitmap read(const std::string& path);
};
