#pragma once

#include <functional>
#include <memory>
#include <string>


class Scene;

// Owns asynchronous MaterialX compilation to realtime shaders for one application session.
class MaterialXSceneRuntime
{
public:
    // onCompleted is called from a worker thread after each compiled program
    // is queued for processPending().
    explicit MaterialXSceneRuntime(std::function<void()> onCompleted = {});
    ~MaterialXSceneRuntime();

    MaterialXSceneRuntime(const MaterialXSceneRuntime&) = delete;
    MaterialXSceneRuntime& operator=(const MaterialXSceneRuntime&) = delete;

    // Stops the worker before owners of completion callbacks are destroyed.
    void shutdown();
    // Processes completions, queues invalidated materials and publishes the
    // compiled programs. Returns true when it published any.
    bool processPending(Scene& scene, const std::string& sceneDirectory = {});
    // Synchronous entry point for CLI/startup code. Waiting is condition-
    // variable based; it never polls futures or sleeps between checks.
    void compileAndWait(Scene& scene, const std::string& sceneDirectory = {});
    bool needsCompilation(const Scene& scene) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
