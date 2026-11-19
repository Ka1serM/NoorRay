#pragma once

#include <cstdint>
#include <expected>
#include <string>
#include <type_traits>
#include <utility>

#include "NoorRHI/Core/Error.h"

namespace NoorRay::Core {

enum class ErrorCode : std::uint8_t {
    InvalidArgument,
    InvalidFile,
    InvalidMaterial,
    InvalidState,
    Unsupported,
    Io,
    Gpu,
};

struct Error {
    ErrorCode code = ErrorCode::InvalidState;
    std::string message;
};

[[nodiscard]] inline std::unexpected<Error> fail(const ErrorCode code, std::string message) {
    return std::unexpected(Error{code, std::move(message)});
}

[[nodiscard]] inline std::unexpected<Error> fail(NoorRHI::Error error) {
    return std::unexpected(Error{ErrorCode::Gpu, std::move(error.message)});
}

template <class T>
[[nodiscard]] std::expected<T, Error> check(std::expected<T, NoorRHI::Error> result) {
    if (!result)
        return fail(std::move(result.error()));
    if constexpr (std::is_void_v<T>)
        return {};
    else
        return std::move(*result);
}

}
