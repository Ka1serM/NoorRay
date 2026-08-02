#pragma once

#include <memory>
#include <string>

class Raytracer;
class Scene;

// Owns asynchronous MaterialX-to-SVM compilation for one application session.
class MaterialXSceneRuntime
{
public:
    MaterialXSceneRuntime();
    ~MaterialXSceneRuntime();

    MaterialXSceneRuntime(const MaterialXSceneRuntime&) = delete;
    MaterialXSceneRuntime& operator=(const MaterialXSceneRuntime&) = delete;

    void compilePending(
        Scene& scene, Raytracer& raytracer, const std::string& sceneDirectory = {});
    bool needsCompilation(const Scene& scene) const;
    bool hasPendingCompilations() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
