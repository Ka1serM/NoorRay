//! The opacity of a Gaussian along a ray, shared by the path tracer and the differentiable renderer, so both see the same image.
static const float gaussianMaxAlpha = 0.999;
//! A Gaussian ends exactly here, not at its proxy, so a peak is never in front of the place where the ray enters the proxy.
static const float gaussianMinAlpha = 1.0 / 255.0;

//! The squared cutoff radius in standard deviations: the response ends at the alpha 1/255, as the proxy instances are scaled.
float gaussianCutoffSquared(float opacity) { return 2.0 * log(max(opacity / gaussianMinAlpha, 1.0)); }

[Differentiable]
float3 rotateByInverse(float4 rotation, float3 v) {
    const float3 axis = -rotation.xyz;
    const float3 t = 2.0 * cross(axis, v);
    return v + rotation.w * t + cross(axis, t);
}

struct GaussianPeak : IDifferentiable {
    float t;
    float alpha;
};

//! The ray is in the space where the Gaussian is the unit normal distribution; its peak is where the ray comes closest to the mean.
[Differentiable]
GaussianPeak findGaussianPeak(float opacity, float3 origin, float3 direction) {
    GaussianPeak peak;
    peak.t = -dot(origin, direction) / dot(direction, direction);
    const float3 closest = origin + direction * peak.t;
    peak.alpha = min(opacity * exp(-0.5 * dot(closest, closest)), gaussianMaxAlpha);
    return peak;
}

//! rotation is a unit quaternion (x, y, z, w); scale is linear and positive.
[Differentiable]
GaussianPeak findGaussianPeak(float3 mean, float4 rotation, float3 scale, float opacity, float3 origin, float3 direction) {
    return findGaussianPeak(opacity, rotateByInverse(rotation, origin - mean) / scale, rotateByInverse(rotation, direction) / scale);
}

//! The sixteen real spherical harmonics of the 3DGS convention; direction is a unit vector from the camera to the Gaussian.
float[16] gaussianShBasis(float3 direction) {
    const float x = direction.x;
    const float y = direction.y;
    const float z = direction.z;
    const float xx = x * x;
    const float yy = y * y;
    const float zz = z * z;
    float basis[16];
    basis[0] = 0.28209479177387814;
    basis[1] = -0.4886025119029199 * y;
    basis[2] = 0.4886025119029199 * z;
    basis[3] = -0.4886025119029199 * x;
    basis[4] = 1.0925484305920792 * x * y;
    basis[5] = -1.0925484305920792 * y * z;
    basis[6] = 0.31539156525252005 * (2.0 * zz - xx - yy);
    basis[7] = -1.0925484305920792 * x * z;
    basis[8] = 0.5462742152960396 * (xx - yy);
    basis[9] = -0.5900435899266435 * y * (3.0 * xx - yy);
    basis[10] = 2.890611442640554 * x * y * z;
    basis[11] = -0.4570457994644658 * y * (4.0 * zz - xx - yy);
    basis[12] = 0.3731763325901154 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy);
    basis[13] = -0.4570457994644658 * x * (4.0 * zz - xx - yy);
    basis[14] = 1.445305721320277 * z * (xx - yy);
    basis[15] = -0.5900435899266435 * x * (xx - 3.0 * yy);
    return basis;
}

[Differentiable]
float srgbToLinear(float encoded) { return encoded <= 0.04045 ? encoded / 12.92 : pow((encoded + 0.055) / 1.055, 2.4); }

//! 3DGS trains the SH on sRGB-encoded images, so its output is display-encoded and becomes linear radiance here.
[Differentiable]
float3 srgbToLinear(float3 encoded) { return float3(srgbToLinear(encoded.x), srgbToLinear(encoded.y), srgbToLinear(encoded.z)); }

[Differentiable]
float3 decodeGaussianColor(float3 shSum) { return srgbToLinear(max(shSum + 0.5, float3(0.0, 0.0, 0.0))); }
