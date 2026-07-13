#include "OrthographicCamera.h"

#include <algorithm>
#include <cmath>

bool OrthographicCamera::renderUi()
{
    const bool cameraChanged = Camera::renderUi();
    const bool sensorChanged = getSensor().renderUi();
    return cameraChanged || sensorChanged;
}
