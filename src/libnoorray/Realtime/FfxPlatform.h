#pragma once

// Builds the FidelityFX SDK (external/FidelityFX-SDK) on Linux. The SDK's
// host code is written against MSVC: it relies on a few *_s string
// functions and on standard headers that MSVC includes transitively. This
// header is force-included into the SDK sources and included first by any
// NoorRay file that uses SDK headers, so both sides see identical types.

#include <bit>
#include <cmath>
#include <codecvt>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <locale>
#include <string>

#ifndef _countof
template<class T, std::size_t N>
constexpr std::size_t ffxPlatformCountOf(const T (&)[N]) noexcept { return N; }
#define _countof(array) ffxPlatformCountOf(array)
#endif

inline int wcscpy_s(wchar_t* destination, const std::size_t size, const wchar_t* source)
{
    if (!destination || size == 0)
        return EINVAL;
    std::wcsncpy(destination, source ? source : L"", size - 1);
    destination[size - 1] = L'\0';
    return 0;
}

template<std::size_t N>
inline int wcscpy_s(wchar_t (&destination)[N], const wchar_t* source)
{
    return wcscpy_s(destination, N, source);
}

inline int strcpy_s(char* destination, const std::size_t size, const char* source)
{
    if (!destination || size == 0)
        return EINVAL;
    std::strncpy(destination, source ? source : "", size - 1);
    destination[size - 1] = '\0';
    return 0;
}

inline int wcstombs_s(std::size_t* converted, char* destination, const std::size_t size,
    const wchar_t* source, const std::size_t count)
{
    if (!destination || size == 0)
        return EINVAL;
    const std::size_t limit = count < size ? count : size - 1;
    std::size_t written = source ? std::wcstombs(destination, source, limit) : 0;
    if (written == static_cast<std::size_t>(-1))
        written = 0;
    destination[written < size ? written : size - 1] = '\0';
    if (converted)
        *converted = written + 1;
    return 0;
}

template<std::size_t N>
inline int sprintf_s(char (&destination)[N], const char* format, ...)
{
    va_list arguments;
    va_start(arguments, format);
    const int result = std::vsnprintf(destination, N, format, arguments);
    va_end(arguments);
    return result;
}

inline int sprintf_s(char* destination, const std::size_t size, const char* format, ...)
{
    va_list arguments;
    va_start(arguments, format);
    const int result = std::vsnprintf(destination, size, format, arguments);
    va_end(arguments);
    return result;
}

// The SDK sizes its opaque contexts for Windows, where wchar_t is 2 bytes.
// Linux's 4-byte wchar_t grows the private FSR context past that size, so
// enlarge the default. ffx_types.h is include-guarded and the context sizes
// expand this macro where they are used, so every translation unit agrees.
#include <FidelityFX/host/ffx_types.h>
#include <FidelityFX/host/ffx_util.h>
#undef FFX_SDK_DEFAULT_CONTEXT_SIZE
#define FFX_SDK_DEFAULT_CONTEXT_SIZE (1024 * 256)
