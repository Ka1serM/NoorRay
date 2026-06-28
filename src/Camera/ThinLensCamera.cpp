#include "ThinLensCamera.h"

#include <cmath>
#include "Camera/CameraInstance.h"

void ThinLensCamera::renderUi(CameraInstance& inst)
{
    Camera::renderUi(inst, true);
    sensor.renderUi(inst);
}
