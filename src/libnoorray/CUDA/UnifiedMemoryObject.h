#pragma once

#include <cstddef>

namespace nr {

class UnifiedMemoryObject {
public:
    static void* operator new(std::size_t size);
    static void* operator new(std::size_t, void* pointer) noexcept { return pointer; }
    static void operator delete(void* pointer) noexcept;
    static void operator delete(void* pointer, std::size_t) noexcept;
    static void operator delete(void*, void*) noexcept {}
};

}
