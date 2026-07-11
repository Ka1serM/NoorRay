#include "RenderSession.h"

#include <stdexcept>

#include "Camera/Camera.h"
#include "Camera/CameraInstance.h"
#include "Camera/PerspectiveCamera.h"
#include "CUDA/rstd/Allocator.h"
#include "Log.h"
#include "Raytracing/Raytracer.h"
#include "Mesh/GaussianAsset.h"
#include "Scene/SceneImporter.h"
#include "Scene/SceneReader.h"

namespace noorray
{

RenderSession::RenderSession(const uint32_t width, const uint32_t height)
    : context(1, 1, true)
    , scene(context)
    , requestedWidth(width)
    , requestedHeight(height)
{
}

RenderSession::~RenderSession() = default;

void RenderSession::loadScene(const std::string& scenePath)
{
    if (SceneImporter::IsSceneFile(scenePath))
    {
        SceneReader::Read(scene, scenePath);
    }
    else
    {
        SceneImporter::ImportFile(scene, scenePath);

        if (!scene.getActiveCamera())
        {
            nr::rstd::allocator<PerspectiveCamera> allocator;
            PerspectiveCamera* camera = allocator.allocate(1);
            allocator.construct(camera);
            Camera cameraHandle(camera);
            cameraHandle.setFocalLength(50.0f);
            scene.add(std::make_unique<CameraInstance>(
                scene, "Camera", Transform{glm::vec3(0.0f, 2.0f, 5.0f)}, cameraHandle));
        }
    }

    LOG_INFO("Scene loaded: " << scenePath << " (" << scene.getGaussianCount()
             << " gaussians, " << scene.getMeshAssets().size() << " meshes)");
    raytracer = std::make_unique<Raytracer>(context, scene);
}

Bitmap RenderSession::renderBitmap(const uint32_t spp)
{
    if (!raytracer)
        throw std::runtime_error("No scene loaded");
    raytracer->setAovEnabled(false);
    return raytracer->renderOffline(spp);
}

void RenderSession::renderToDevice(float* rgbaDevice, const uint32_t spp)
{
    if (!raytracer)
        throw std::runtime_error("No scene loaded");
    raytracer->setAovEnabled(false);
    raytracer->renderOfflineToDevice(rgbaDevice, spp);
}

void RenderSession::setGaussianOpacity(uint32_t gaussianIndex, const float opacity)
{
    for (GaussianAsset& asset : scene.getGaussianAssets())
    {
        if (gaussianIndex < asset.getGaussianCount())
        {
            asset.getGaussians()[gaussianIndex].opacity = opacity;
            if (raytracer)
                raytracer->updateMeshes();
            return;
        }
        gaussianIndex -= asset.getGaussianCount();
    }
    throw std::out_of_range("Gaussian index is out of range");
}

uint32_t RenderSession::width() const
{
    return raytracer ? raytracer->getWidth() : requestedWidth;
}

uint32_t RenderSession::height() const
{
    return raytracer ? raytracer->getHeight() : requestedHeight;
}

uint32_t RenderSession::gaussianCount() const
{
    return scene.getGaussianCount();
}

}
