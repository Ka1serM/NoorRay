import NoorRHI;

//! Planck's law in W / (m^2 sr nm) for a wavelength in nanometres.
float planck(float lambdaNm, float kelvin) {
    if (kelvin <= 0.0)
        return 0.0;
    const float c1 = 1.191042972e20;
    const float c2 = 1.4387769e7;
    const float l = lambdaNm;
    return c1 / (l * l * l * l * l * (exp(c2 / (l * kelvin)) - 1.0));
}

//! Planck normalised to one at its peak wavelength (Wien), so that temperature does not change brightness much.
float normalizedPlanck(float lambdaNm, float kelvin) {
    const float peak = 2.8977721e6 / kelvin;
    return planck(lambdaNm, kelvin) / planck(peak, kelvin);
}
