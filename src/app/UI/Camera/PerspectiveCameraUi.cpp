#include "UI/Camera/CameraUi.h"

#include "Camera/PerspectiveCamera.h"

bool camera_ui::render(PerspectiveCamera& camera)
{
    const bool cameraChanged = renderCommon(camera);
    const bool sensorChanged = render(camera.getSensor());
    return cameraChanged || sensorChanged;
}
