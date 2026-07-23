#include "CUDA/Unique/Handle.h"

#include <catch2/catch_test_macros.hpp>

namespace
{
int destroyedHandle{};
int destroyTestHandle(const int handle)
{
    destroyedHandle = handle;
    return 0;
}
}

TEST_CASE("unique CUDA handle transfers and releases ownership", "[cuda][raii]")
{
    destroyedHandle = 0;
    {
        nr::cuda::UniqueHandle<int, destroyTestHandle> first(7);
        auto second = std::move(first);
        CHECK_FALSE(first);
        CHECK(second.get() == 7);
        second.reset(9);
        CHECK(destroyedHandle == 7);
    }
    CHECK(destroyedHandle == 9);
}
