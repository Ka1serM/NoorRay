#pragma once

#include "memory.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <vector>

namespace gpu {

// CPU data plus one GPU allocation. The owner calls commit() because it just
// changed the data, so there is nothing to re-compare first.
template<class T>
class Shared {
public:
    T data{};
    Shared() = default;
    explicit Shared(Device& device) : buffer_(device.buffer<T>(1)) {}
    Shared(const Shared&) = delete;
    Shared& operator=(const Shared&) = delete;
    Shared(Shared&&) noexcept = default;
    Shared& operator=(Shared&&) noexcept = default;

    GpuPtr<T> ptr() const { return buffer_.ptr(); }
    explicit operator bool() const noexcept { return bool(buffer_); }

    // Attaches GPU storage while leaving `data` untouched. Assigning a freshly
    // constructed Shared would reset the struct, which loses anything authored
    // before a device existed - the common case for a record that doubles as
    // the object's own state.
    void allocate(Device& device) { buffer_ = device.buffer<T>(1); }

    // Drops the GPU storage and keeps `data`, so the record can be republished
    // onto a new device later.
    void release() noexcept { buffer_ = {}; }

    void commit() {
        if (!buffer_) throw Error(ErrorCode::InvalidResource, "empty shared record");
        buffer_.upload(std::span<const T>(&data, 1));
    }
private:
    Buffer<T> buffer_;
};

} // namespace gpu
