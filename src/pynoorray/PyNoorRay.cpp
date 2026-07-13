#include <nanobind/nanobind.h>
#include "Bindings.h"

namespace nb = nanobind;

NB_MODULE(_pynoorray, module)
{
    bindVector3(module);
    bindSellmeierCoefficients(module);
    bindTransform(module);
    bindMaterial(module);
    bindRenderSettings(module);
    bindTexture(module);
    bindEnvironment(module);
    bindSceneObject(module);
    bindMeshAsset(module);
    bindMeshInstance(module);
    bindSensor(module);
    bindRectangularSensor(module);
    bindScatterPsfSensor(module);
    bindGatherPsfSensor(module);
    bindCamera(module);
    bindPerspectiveCamera(module);
    bindThinLensCamera(module);
    bindOrthographicCamera(module);
    bindFisheyeCamera(module);
    bindRealisticCamera(module);
    bindHybridPsfCamera(module);
    bindCameraInstance(module);
    bindScene(module);
    bindContext(module);
    bindImage(module);
    bindRaytracer(module);
    bindNoorRaySession(module);
}
