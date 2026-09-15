#pragma once

#include <memory>
#include <string>


class Scene;

// Owns asynchronous MaterialX-to-SVM compilation for one application session.
class MaterialXSceneRuntime
{
public:
    MaterialXSceneRuntime();
    ~MaterialXSceneRuntime();

    MaterialXSceneRuntime(const MaterialXSceneRuntime&) = delete;
    MaterialXSceneRuntime& operator=(const MaterialXSceneRuntime&) = delete;

    // Stops the worker before owners of completion callbacks are destroyed.
    void shutdown();
    // Processes completions and queues invalidated materials. This is called
    // from explicit material-change notifications, not from the render loop.
    void processPending(Scene& scene, const std::string& sceneDirectory = {});
    // Synchronous entry point for CLI/startup code. Waiting is condition-
    // variable based; it never polls futures or sleeps between checks.
    void compileAndWait(Scene& scene, const std::string& sceneDirectory = {});
    bool needsCompilation(const Scene& scene) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
