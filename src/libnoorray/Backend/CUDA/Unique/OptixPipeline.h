#pragma once

#include <utility>

#include <optix.h>
#include <optix_stubs.h>

namespace nr::cuda
{

class UniqueOptixPipeline
{
public:
    UniqueOptixPipeline() = default;
    ~UniqueOptixPipeline() noexcept { reset(); }

    UniqueOptixPipeline(const UniqueOptixPipeline&) = delete;
    UniqueOptixPipeline& operator=(const UniqueOptixPipeline&) = delete;

    UniqueOptixPipeline(UniqueOptixPipeline&& other) noexcept
        : pipeline(std::exchange(other.pipeline, nullptr))
    {
    }

    UniqueOptixPipeline& operator=(UniqueOptixPipeline&& other) noexcept
    {
        if (this != &other)
        {
            reset();
            pipeline = std::exchange(other.pipeline, nullptr);
        }
        return *this;
    }

    void reset(OptixPipeline newPipeline = nullptr) noexcept
    {
        if (pipeline != nullptr)
            optixPipelineDestroy(pipeline);
        pipeline = newPipeline;
    }

    OptixPipeline get() const noexcept { return pipeline; }
    OptixPipeline* put() noexcept
    {
        reset();
        return &pipeline;
    }
    explicit operator bool() const noexcept { return pipeline != nullptr; }

private:
    OptixPipeline pipeline{};
};

}
