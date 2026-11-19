#include "NoorRay/Core/Log.h"

#include <cstdio>
#include <mutex>

namespace NoorRay::Core {

namespace {

constexpr std::string_view levelNames[] = {"debug", "info", "warning", "error"};

struct LogState {
    std::mutex mutex;
    LogSink sink;
    LogLevel level = LogLevel::Info;
};

LogState& state() {
    static LogState instance;
    return instance;
}

}

void Log::setSink(LogSink sink) {
    std::scoped_lock lock(state().mutex);
    state().sink = std::move(sink);
}

void Log::setLevel(const LogLevel level) {
    std::scoped_lock lock(state().mutex);
    state().level = level;
}

void Log::write(const LogLevel level, const std::string_view message) {
    std::scoped_lock lock(state().mutex);
    if (level < state().level)
        return;
    if (state().sink) {
        state().sink(level, message);
        return;
    }
    const std::string_view name = levelNames[static_cast<std::size_t>(level)];
    std::fprintf(stderr, "[NoorRay %.*s] %.*s\n", static_cast<int>(name.size()), name.data(), static_cast<int>(message.size()), message.data());
}

}
