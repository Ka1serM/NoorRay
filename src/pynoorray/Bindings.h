#pragma once

#include <nanobind/nanobind.h>

void bindVector3(nanobind::module_& module);
void bindSellmeierCoefficients(nanobind::module_& module);
void bindTransform(nanobind::module_& module);
void bindMaterial(nanobind::module_& module);
void bindRenderSettings(nanobind::module_& module);
void bindTexture(nanobind::module_& module);
void bindEnvironment(nanobind::module_& module);
void bindSceneObject(nanobind::module_& module);
void bindMeshAsset(nanobind::module_& module);
void bindMeshInstance(nanobind::module_& module);
void bindSensor(nanobind::module_& module);
void bindRectangularSensor(nanobind::module_& module);
void bindScatterPsfSensor(nanobind::module_& module);
void bindGatherPsfSensor(nanobind::module_& module);
void bindCamera(nanobind::module_& module);
void bindPerspectiveCamera(nanobind::module_& module);
void bindThinLensCamera(nanobind::module_& module);
void bindOrthographicCamera(nanobind::module_& module);
void bindFisheyeCamera(nanobind::module_& module);
void bindRealisticCamera(nanobind::module_& module);
void bindHybridPsfCamera(nanobind::module_& module);
void bindCameraInstance(nanobind::module_& module);
void bindScene(nanobind::module_& module);
void bindContext(nanobind::module_& module);
void bindImage(nanobind::module_& module);
void bindRaytracer(nanobind::module_& module);
void bindNoorRaySession(nanobind::module_& module);
