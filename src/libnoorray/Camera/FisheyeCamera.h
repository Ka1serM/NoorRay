#pragma once

#include "Camera/Camera.h"

class FisheyeCamera : public Camera {
public:
    FisheyeCamera();
    explicit FisheyeCamera(std::unique_ptr<Sensor> sensor);
    FisheyeCamera(const FisheyeCamera& other);
    ~FisheyeCamera();
    float apertureDiameterMm{};
    float bokehBias{1.f};

    bool renderUi();
};
