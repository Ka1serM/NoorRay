#pragma once

#include <cstdint>
#include <format>
#include <functional>
#include <string_view>
#include <utility>

namespace NoorRay::Core {

enum class LogLevel : std::uint8_t { Debug, Info, Warning, Error };

using LogSink = std::function<void(LogLevel level, std::string_view message)>;

class Log {
public:
    static void setSink(LogSink sink);
    static void setLevel(LogLevel level);
    static void write(LogLevel level, std::string_view message);

    template <class... Arguments>
    static void debug(std::format_string<Arguments...> format, Arguments&&... arguments) {
        write(LogLevel::Debug, std::format(format, std::forward<Arguments>(arguments)...));
    }
    template <class... Arguments>
    static void info(std::format_string<Arguments...> format, Arguments&&... arguments) {
        write(LogLevel::Info, std::format(format, std::forward<Arguments>(arguments)...));
    }
    template <class... Arguments>
    static void warning(std::format_string<Arguments...> format, Arguments&&... arguments) {
        write(LogLevel::Warning, std::format(format, std::forward<Arguments>(arguments)...));
    }
    template <class... Arguments>
    static void error(std::format_string<Arguments...> format, Arguments&&... arguments) {
        write(LogLevel::Error, std::format(format, std::forward<Arguments>(arguments)...));
    }
};

}
