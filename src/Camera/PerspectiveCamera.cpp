#include "PerspectiveCamera.h"

#include <cmath>
#include "Camera/CameraInstance.h"

void PerspectiveCamera::renderUi(CameraInstance& inst)
{
    Camera::renderUi(inst, false);
    sensor.renderUi(inst);
}
