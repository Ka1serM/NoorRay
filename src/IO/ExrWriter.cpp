#include "IO/ExrWriter.h"

#include <cstdlib>

#include <tinyexr.h>

bool writeFloatExr(
    const std::string& path, const float* rgba, const uint32_t width, const uint32_t height,
    std::string* errorMessage)
{
    const char* error = nullptr;
    // save_as_fp16 = 0 is required: preserve all four input channels as FP32.
    const int result = SaveEXR(
        rgba, static_cast<int>(width), static_cast<int>(height), 4, 0,
        path.c_str(), &error);
    if (result == TINYEXR_SUCCESS)
        return true;
    if (errorMessage != nullptr)
        *errorMessage = error != nullptr ? error : "Unknown TinyEXR error";
    if (error != nullptr)
        FreeEXRErrorMessage(error);
    return false;
}
