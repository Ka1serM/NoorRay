#pragma once

//! Force-included into the FidelityFX sources and included first by NoorRay files that use FidelityFX, so that both sides see the same types.

#include <bit>
#include <cerrno>
#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cwchar>

#if !defined(_WIN32)
template <class T, std::size_t N>
constexpr std::size_t ffxCountOf(const T (&)[N]) noexcept {
    return N;
}
#define _countof(array) ffxCountOf(array)

inline int wcscpy_s(wchar_t* destination, const std::size_t size, const wchar_t* source) {
    if (!destination || size == 0)
        return EINVAL;
    std::wcsncpy(destination, source ? source : L"", size - 1);
    destination[size - 1] = L'\0';
    return 0;
}

template <std::size_t N>
inline int wcscpy_s(wchar_t (&destination)[N], const wchar_t* source) {
    return wcscpy_s(destination, N, source);
}

inline int strcpy_s(char* destination, const std::size_t size, const char* source) {
    if (!destination || size == 0)
        return EINVAL;
    std::strncpy(destination, source ? source : "", size - 1);
    destination[size - 1] = '\0';
    return 0;
}

inline int sprintf_s(char* destination, const std::size_t size, const char* format, ...) {
    va_list arguments;
    va_start(arguments, format);
    const int result = std::vsnprintf(destination, size, format, arguments);
    va_end(arguments);
    return result;
}

inline int swprintf_s(wchar_t* destination, const std::size_t size, const wchar_t* format, ...) {
    va_list arguments;
    va_start(arguments, format);
    const int result = std::vswprintf(destination, size, format, arguments);
    va_end(arguments);
    return result;
}

//! The SDK sizes its opaque contexts for the 2-byte wchar_t of Windows, so the 4-byte wchar_t elsewhere needs larger ones.
#include <FidelityFX/host/ffx_types.h>
#include <FidelityFX/host/ffx_util.h>
#undef FFX_SDK_DEFAULT_CONTEXT_SIZE
#define FFX_SDK_DEFAULT_CONTEXT_SIZE (1024 * 256)
#endif
