static const uint shCount = 16;
static const float dilation = 0.3;
static const float frustumSlack = 1.3;
static const float shC0 = 0.28209479177387814;
static const float shC1 = 0.4886025119029199;
static const float shC2[5] = { 1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396 };
static const float shC3[7] = { -0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
                               -0.4570457994644658, 1.445305721320277, -0.5900435899266435 };

struct ProjectionView {
    float3x3 rotation;
    float3 translation;
    float4 intrinsics;
    float3 position;
    float2 tanFov;
}

struct GaussianInput : IDifferentiable {
    float3 mean;
    float3 logScale;
    float4 rotation;
    float logitOpacity;
    float3 sh[shCount];
}

struct Splat : IDifferentiable {
    float2 mean;
    float3 conic;
    float opacity;
    float3 color;
}

[Differentiable]
float3x3 getRotationMatrix(float4 quaternion) {
    const float4 q = quaternion / length(quaternion);
    const float r = q.x;
    const float x = q.y;
    const float y = q.z;
    const float z = q.w;
    return float3x3(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - r * z), 2.0 * (x * z + r * y),
                    2.0 * (x * y + r * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - r * x),
                    2.0 * (x * z - r * y), 2.0 * (y * z + r * x), 1.0 - 2.0 * (x * x + y * y));
}

[Differentiable]
float3 getCovariance2d(float3 mean, float3 logScale, float4 rotation, no_diff ProjectionView view) {
    const float3x3 r = getRotationMatrix(rotation);
    const float3 scale = exp(logScale);
    const float3x3 m = float3x3(r[0] * scale, r[1] * scale, r[2] * scale);
    const float3x3 covariance = mul(view.rotation, mul(mul(m, transpose(m)), transpose(view.rotation)));
    const float3 t = mul(view.rotation, mean) + view.translation;
    const float2 limit = frustumSlack * view.tanFov;
    const float2 clamped = min(limit, max(-limit, t.xy / t.z)) * t.z;
    const float4 k = view.intrinsics;
    const float3 jx = float3(k.x / t.z, 0.0, -k.x * clamped.x / (t.z * t.z));
    const float3 jy = float3(0.0, k.y / t.z, -k.y * clamped.y / (t.z * t.z));
    const float3 cx = mul(covariance, jx);
    const float3 cy = mul(covariance, jy);
    return float3(dot(jx, cx) + dilation, dot(jx, cy), dot(jy, cy) + dilation);
}

[Differentiable]
float2 getMean2d(float3 mean, no_diff ProjectionView view) {
    const float3 t = mul(view.rotation, mean) + view.translation;
    return view.intrinsics.xy * t.xy / t.z + view.intrinsics.zw - 0.5;
}

[Differentiable]
float3 getColor(float3 mean, float3 sh[shCount], no_diff float3 cameraPosition, no_diff uint degree) {
    const float3 direction = normalize(mean - cameraPosition);
    const float x = direction.x;
    const float y = direction.y;
    const float z = direction.z;
    float3 result = shC0 * sh[0];
    if (degree > 0)
        result = result - shC1 * y * sh[1] + shC1 * z * sh[2] - shC1 * x * sh[3];
    if (degree > 1) {
        const float xx = x * x;
        const float yy = y * y;
        const float zz = z * z;
        result = result + shC2[0] * x * y * sh[4] + shC2[1] * y * z * sh[5] + shC2[2] * (2.0 * zz - xx - yy) * sh[6] + shC2[3] * x * z * sh[7]
                 + shC2[4] * (xx - yy) * sh[8];
        if (degree > 2)
            result = result + shC3[0] * y * (3.0 * xx - yy) * sh[9] + shC3[1] * x * y * z * sh[10] + shC3[2] * y * (4.0 * zz - xx - yy) * sh[11]
                     + shC3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh[12] + shC3[4] * x * (4.0 * zz - xx - yy) * sh[13]
                     + shC3[5] * z * (xx - yy) * sh[14] + shC3[6] * x * (xx - 3.0 * yy) * sh[15];
    }
    return max(result + 0.5, float3(0.0));
}

[Differentiable]
Splat getSplat(GaussianInput gaussian, no_diff ProjectionView view, no_diff uint degree) {
    const float3 covariance = getCovariance2d(gaussian.mean, gaussian.logScale, gaussian.rotation, view);
    const float determinant = covariance.x * covariance.z - covariance.y * covariance.y;
    Splat splat;
    splat.mean = getMean2d(gaussian.mean, view);
    splat.conic = float3(covariance.z, -covariance.y, covariance.x) / determinant;
    splat.opacity = 1.0 / (1.0 + exp(-gaussian.logitOpacity));
    splat.color = getColor(gaussian.mean, gaussian.sh, view.position, degree);
    return splat;
}
