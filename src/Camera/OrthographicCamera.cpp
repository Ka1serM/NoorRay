#include "OrthographicCamera.h"

#include <algorithm>
#include <cmath>
#include "Camera/CameraInstance.h"

void OrthographicCamera::renderUi(CameraInstance& inst)
{
    Camera::renderUi(inst, false);
    sensor.renderUi(inst);
}
