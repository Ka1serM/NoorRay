#include "PerspectiveCamera.h"

#include <cmath>

bool PerspectiveCamera::renderUi()
{
    const bool cameraChanged = Camera::renderUi();
    const bool sensorChanged = getSensor().renderUi();
    return cameraChanged || sensorChanged;
}
