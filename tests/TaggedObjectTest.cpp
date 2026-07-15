#include <catch2/catch_test_macros.hpp>

#include <cuda_runtime.h>

#include <memory>

#include "CUDA/TaggedPointer.h"

namespace {

class Operation;
class Add;
class Multiply;

using TaggedOperation = nr::TaggedObject<Operation, Add, Multiply>;

class Operation : public TaggedOperation {
public:
    explicit Operation(int operand) : operand(operand) {}
    virtual ~Operation() = default;
    virtual int applyOnCpu(int value) const = 0;

protected:
    int operand{};
};

class Add : public Operation::Type<Add> {
public:
    explicit Add(int operand) : TaggedBase(operand) {}
    int apply(int value) const { return value + operand; }
    int applyOnCpu(int value) const override { return apply(value); }
};

class Multiply : public Operation::Type<Multiply> {
public:
    explicit Multiply(int operand) : TaggedBase(operand) {}
    int apply(int value) const { return value * operand; }
    int applyOnCpu(int value) const override { return apply(value); }
};

}

TEST_CASE("ordinary unique_ptr construction creates a managed tagged object", "[cuda][tagged]")
{
    std::unique_ptr<Operation> operation = std::make_unique<Multiply>(3);

    REQUIRE(operation->Is<Multiply>());
    REQUIRE_FALSE(operation->Is<Add>());
    CHECK(operation->applyOnCpu(7) == 21);
    CHECK(operation->Dispatch([](const auto* concrete) { return concrete->apply(7); }) == 21);

    cudaPointerAttributes attributes{};
    REQUIRE(cudaPointerGetAttributes(&attributes, operation.get()) == cudaSuccess);
    CHECK(attributes.type == cudaMemoryTypeManaged);
}

TEST_CASE("stack construction automatically records the concrete type", "[tagged]")
{
    Add operation(5);
    Operation& base = operation;

    REQUIRE(base.Is<Add>());
    REQUIRE_FALSE(base.Is<Multiply>());
    CHECK(base.Dispatch([](const auto* concrete) { return concrete->apply(7); }) == 12);
}
