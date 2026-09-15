#pragma once

class Camera;
class Sensor;
class PerspectiveCamera;
class ThinLensCamera;
class OrthographicCamera;
class FisheyeCamera;
class RealisticCamera;

// Editor property rows for cameras and sensors. These live in the app, not in
// libnoorray: the renderer's camera types carry no UI or file-dialog state.
// Each function returns true when it changed the object.
namespace camera_ui
{
// Rows shared by every projection (exposure, field of view, focal length).
bool renderCommon(Camera& camera);
bool render(Sensor& sensor);
bool render(PerspectiveCamera& camera);
bool render(ThinLensCamera& camera);
bool render(OrthographicCamera& camera);
bool render(FisheyeCamera& camera);
bool render(RealisticCamera& camera);
}
