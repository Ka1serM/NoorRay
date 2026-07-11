#include "PerspectiveCamera.h"

#include <cmath>

bool PerspectiveCamera::renderUi()
{
    const bool cameraChanged = Camera::renderUi();
    const bool sensorChanged = sensor.renderUi();
    return cameraChanged || sensorChanged;
}
