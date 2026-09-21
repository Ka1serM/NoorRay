#pragma once

#include "Camera/Camera.h"

class FisheyeCamera : public Camera {
public:
    FisheyeCamera();
    explicit FisheyeCamera(std::unique_ptr<Sensor> sensor);
    FisheyeCamera(const FisheyeCamera& other);
    ~FisheyeCamera();
    float getApertureDiameterMm() const override { return apertureDiameterMm; }
    void setApertureDiameterMm(const float value) override
    {
        apertureDiameterMm = std::max(0.0f, value);
        data.apertureDiameterMm = apertureDiameterMm;
        notifyChanged();
    }
    float getBokehBias() const override { return bokehBias; }
    void setBokehBias(const float value) override { bokehBias = std::max(0.001f, value); notifyChanged(); }

private:
    float apertureDiameterMm{};
    float bokehBias{1.f};
};
