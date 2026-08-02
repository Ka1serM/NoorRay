#include "Rendering/Camera/OrthographicCamera.h"

bool OrthographicCamera::renderUi()
{
    const bool cameraChanged = Camera::renderUi();
    const bool sensorChanged = getSensor().renderUi();
    return cameraChanged || sensorChanged;
}
