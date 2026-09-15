#include "UI/Camera/CameraUi.h"

#include "Camera/OrthographicCamera.h"

bool camera_ui::render(OrthographicCamera& camera)
{
    const bool cameraChanged = renderCommon(camera);
    const bool sensorChanged = render(camera.getSensor());
    return cameraChanged || sensorChanged;
}
