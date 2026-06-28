#include "FisheyeCamera.h"

#include <algorithm>
#include <cmath>
#include "Camera/CameraInstance.h"

void FisheyeCamera::renderUi(CameraInstance& inst)
{
    Camera::renderUi(inst, true);
    sensor.renderUi(inst);
}
