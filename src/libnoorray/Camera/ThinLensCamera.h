#pragma once

#include "Camera/Camera.h"

class ThinLensCamera : public Camera {
public:
    ThinLensCamera();
    explicit ThinLensCamera(std::unique_ptr<Sensor> sensor);
    ThinLensCamera(const ThinLensCamera& other);
    ~ThinLensCamera();
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
