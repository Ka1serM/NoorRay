#include "NoorRay.OpenDlssRr/Http.h"

#include <cstring>
#include <format>
#include <string>

#ifdef _WIN32
#include <windows.h>
#include <winhttp.h>
#else
#include <dlfcn.h>
#endif

namespace NoorRay::OpenDlssRr {

using Core::Error;
using Core::ErrorCode;
using Core::fail;

#ifdef _WIN32

namespace {

struct Handle {
    HINTERNET handle = nullptr;
    ~Handle() {
        if (handle)
            WinHttpCloseHandle(handle);
    }
};

std::unexpected<Error> failLast(const std::string_view url, const std::string_view step) {
    return fail(ErrorCode::Io, std::format("cannot download {}: {} failed with error {}", url, step, GetLastError()));
}

}

std::expected<std::vector<std::byte>, Error> downloadRange(const std::string_view url, const std::uint64_t offset, const std::uint64_t size) {
    const std::wstring address(url.begin(), url.end());
    URL_COMPONENTS parts{};
    parts.dwStructSize = sizeof(parts);
    parts.dwHostNameLength = static_cast<DWORD>(-1);
    parts.dwUrlPathLength = static_cast<DWORD>(-1);
    if (!WinHttpCrackUrl(address.c_str(), 0, 0, &parts))
        return failLast(url, "WinHttpCrackUrl");
    const std::wstring host(parts.lpszHostName, parts.dwHostNameLength);
    const std::wstring path(parts.lpszUrlPath, parts.dwUrlPathLength);
    const Handle session{WinHttpOpen(L"NoorRay", WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0)};
    if (!session.handle)
        return failLast(url, "WinHttpOpen");
    const Handle connection{WinHttpConnect(session.handle, host.c_str(), parts.nPort, 0)};
    if (!connection.handle)
        return failLast(url, "WinHttpConnect");
    const DWORD secure = parts.nScheme == INTERNET_SCHEME_HTTPS ? WINHTTP_FLAG_SECURE : 0;
    const Handle request{WinHttpOpenRequest(connection.handle, L"GET", path.c_str(), nullptr, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, secure)};
    if (!request.handle)
        return failLast(url, "WinHttpOpenRequest");
    const std::string range = std::format("Range: bytes={}-{}", offset, offset + size - 1);
    const std::wstring header(range.begin(), range.end());
    if (!WinHttpSendRequest(request.handle, header.c_str(), static_cast<DWORD>(header.size()), WINHTTP_NO_REQUEST_DATA, 0, 0, 0))
        return failLast(url, "WinHttpSendRequest");
    if (!WinHttpReceiveResponse(request.handle, nullptr))
        return failLast(url, "WinHttpReceiveResponse");
    DWORD status = 0;
    DWORD statusSize = sizeof(status);
    if (!WinHttpQueryHeaders(request.handle, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, WINHTTP_HEADER_NAME_BY_INDEX, &status, &statusSize,
                             WINHTTP_NO_HEADER_INDEX))
        return failLast(url, "WinHttpQueryHeaders");
    if (status != 200 && status != 206)
        return fail(ErrorCode::Io, std::format("cannot download {}: HTTP {}", url, status));
    std::vector<std::byte> bytes;
    bytes.reserve(static_cast<std::size_t>(size));
    const std::size_t limit = static_cast<std::size_t>(offset + size);
    for (;;) {
        std::byte buffer[1 << 16];
        DWORD read = 0;
        if (!WinHttpReadData(request.handle, buffer, sizeof(buffer), &read))
            return failLast(url, "WinHttpReadData");
        if (read == 0)
            break;
        if (bytes.size() + read > limit)
            return fail(ErrorCode::Io, std::format("{} sent more than {} bytes", url, limit));
        bytes.insert(bytes.end(), buffer, buffer + read);
    }
    if (status == 200 && bytes.size() == offset + size)
        bytes.erase(bytes.begin(), bytes.begin() + static_cast<std::ptrdiff_t>(offset));
    if (bytes.size() != size)
        return fail(ErrorCode::Io, std::format("{} sent {} bytes, not {}", url, bytes.size(), size));
    return bytes;
}

#else

namespace {

//! The option numbers of curl/curl.h: NoorRay loads the libcurl of the system at run time, so the build needs no curl headers.
using Curl = void;
constexpr int optionWriteData = 10001;
constexpr int optionUrl = 10002;
constexpr int optionRange = 10007;
constexpr int optionUserAgent = 10018;
constexpr int optionWriteFunction = 20011;
constexpr int optionFailOnError = 45;
constexpr int optionFollowLocation = 52;
constexpr int optionConnectTimeout = 78;
constexpr int optionNoSignal = 99;
constexpr int optionLowSpeedLimit = 19;
constexpr int optionLowSpeedTime = 20;
constexpr int infoResponseCode = 0x200002;
constexpr long globalDefault = 3;

struct CurlLibrary {
    int (*globalInit)(long) = nullptr;
    Curl* (*init)() = nullptr;
    int (*setopt)(Curl*, int, ...) = nullptr;
    int (*perform)(Curl*) = nullptr;
    int (*getinfo)(Curl*, int, ...) = nullptr;
    void (*cleanup)(Curl*) = nullptr;
    const char* (*strerror)(int) = nullptr;
    std::string error;
};

template <class Function>
bool load(void* library, const char* name, Function& function) {
    function = reinterpret_cast<Function>(dlsym(library, name));
    return function != nullptr;
}

//! Never unloaded, because a detached download thread can still use it while the program exits.
const CurlLibrary& getCurl() {
    static const CurlLibrary curl = [] {
        CurlLibrary loaded;
        void* library = nullptr;
        for (const char* name : {"libcurl.so.4", "libcurl-gnutls.so.4", "libcurl.so"})
            if ((library = dlopen(name, RTLD_NOW | RTLD_LOCAL)))
                break;
        if (!library) {
            loaded.error = "libcurl is not installed (libcurl.so.4)";
            return loaded;
        }
        if (!load(library, "curl_global_init", loaded.globalInit) || !load(library, "curl_easy_init", loaded.init)
            || !load(library, "curl_easy_setopt", loaded.setopt) || !load(library, "curl_easy_perform", loaded.perform)
            || !load(library, "curl_easy_getinfo", loaded.getinfo) || !load(library, "curl_easy_cleanup", loaded.cleanup)
            || !load(library, "curl_easy_strerror", loaded.strerror)) {
            loaded = {};
            loaded.error = "the libcurl of the system lacks a function";
            return loaded;
        }
        loaded.globalInit(globalDefault);
        return loaded;
    }();
    return curl;
}

struct Receiver {
    std::vector<std::byte> bytes;
    std::size_t limit = 0;
};

std::size_t receive(const char* data, const std::size_t size, const std::size_t count, void* user) {
    auto& receiver = *static_cast<Receiver*>(user);
    const std::size_t length = size * count;
    if (receiver.bytes.size() + length > receiver.limit)
        return 0;
    const auto* first = reinterpret_cast<const std::byte*>(data);
    receiver.bytes.insert(receiver.bytes.end(), first, first + length);
    return length;
}

}

std::expected<std::vector<std::byte>, Error> downloadRange(const std::string_view url, const std::uint64_t offset, const std::uint64_t size) {
    const CurlLibrary& curl = getCurl();
    if (!curl.init)
        return fail(ErrorCode::Unsupported, curl.error);
    Curl* handle = curl.init();
    if (!handle)
        return fail(ErrorCode::Io, "libcurl cannot start a download");
    const std::string address(url);
    const std::string range = std::format("{}-{}", offset, offset + size - 1);
    //! A server that ignores the Range header answers 200 with the whole file.
    Receiver receiver{.limit = static_cast<std::size_t>(offset + size)};
    receiver.bytes.reserve(static_cast<std::size_t>(size));
    curl.setopt(handle, optionUrl, address.c_str());
    curl.setopt(handle, optionRange, range.c_str());
    curl.setopt(handle, optionUserAgent, "NoorRay");
    curl.setopt(handle, optionWriteFunction, &receive);
    curl.setopt(handle, optionWriteData, &receiver);
    curl.setopt(handle, optionFailOnError, 1L);
    curl.setopt(handle, optionFollowLocation, 1L);
    curl.setopt(handle, optionNoSignal, 1L);
    curl.setopt(handle, optionConnectTimeout, 30L);
    curl.setopt(handle, optionLowSpeedLimit, 1024L);
    curl.setopt(handle, optionLowSpeedTime, 60L);
    const int result = curl.perform(handle);
    long status = 0;
    curl.getinfo(handle, infoResponseCode, &status);
    curl.cleanup(handle);
    if (result != 0)
        return fail(ErrorCode::Io, std::format("cannot download {}: {}", url, curl.strerror(result)));
    if (status == 200 && receiver.bytes.size() == offset + size)
        receiver.bytes.erase(receiver.bytes.begin(), receiver.bytes.begin() + static_cast<std::ptrdiff_t>(offset));
    if (receiver.bytes.size() != size)
        return fail(ErrorCode::Io, std::format("{} sent {} bytes, not {}", url, receiver.bytes.size(), size));
    return std::move(receiver.bytes);
}

#endif

}
