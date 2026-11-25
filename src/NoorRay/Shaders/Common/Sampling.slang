import NoorRHI;
import SceneRecords;
import Common.Math;

float2 sampleConcentricDisk(float2 u) {
    const float2 offset = 2.0 * u - 1.0;
    if (offset.x == 0.0 && offset.y == 0.0)
        return float2(0.0, 0.0);
    float radius;
    float theta;
    if (abs(offset.x) > abs(offset.y)) {
        radius = offset.x;
        theta = (pi / 4.0) * (offset.y / offset.x);
    } else {
        radius = offset.y;
        theta = pi / 2.0 - (pi / 4.0) * (offset.x / offset.y);
    }
    return radius * float2(cos(theta), sin(theta));
}

//! A regular polygon aperture with the given number of blades; fewer than three blades is a disk.
float2 samplePolygon(float2 u, uint blades, float rotation) {
    if (blades < 3)
        return sampleConcentricDisk(u);
    const float sector = twoPi / float(blades);
    const float scaled = u.x * float(blades);
    const uint index = min(uint(scaled), blades - 1);
    const float a = sqrt(u.y);
    const float b = scaled - float(index);
    const float angle0 = rotation + sector * float(index);
    const float2 p0 = float2(cos(angle0), sin(angle0));
    const float2 p1 = float2(cos(angle0 + sector), sin(angle0 + sector));
    return a * ((1.0 - b) * p0 + b * p1);
}

float3 sampleCosineHemisphere(float2 u) {
    const float2 disk = sampleConcentricDisk(u);
    return float3(disk, safeSqrt(1.0 - dot(disk, disk)));
}

float cosineHemispherePdf(float cosTheta) { return max(cosTheta, 0.0) * invPi; }

float3 sampleUniformSphere(float2 u) {
    const float z = 1.0 - 2.0 * u.x;
    const float r = safeSqrt(1.0 - z * z);
    const float phi = twoPi * u.y;
    return float3(r * cos(phi), r * sin(phi), z);
}

//! Directions inside the cone around +Z with the given cosine.
//! The u of sampleConcentricDisk that gives the point p of the unit disk.
float2 inverseConcentricDisk(float2 p) {
    const float radius = length(p);
    float phi = atan2(p.y, p.x);
    if (phi < -pi / 4.0)
        phi += twoPi;
    float2 offset;
    if (phi < pi / 4.0)
        offset = float2(radius, phi * radius / (pi / 4.0));
    else if (phi < 3.0 * pi / 4.0)
        offset = float2(-(phi - pi / 2.0) * radius / (pi / 4.0), radius);
    else if (phi < 5.0 * pi / 4.0)
        offset = float2(-radius, -(phi - pi) * radius / (pi / 4.0));
    else
        offset = float2((phi - 3.0 * pi / 2.0) * radius / (pi / 4.0), -radius);
    return 0.5 + 0.5 * offset;
}

float3 sampleUniformCone(float2 u, float cosThetaMax) {
    const float cosTheta = 1.0 - u.x * (1.0 - cosThetaMax);
    const float sinTheta = safeSqrt(1.0 - cosTheta * cosTheta);
    const float phi = twoPi * u.y;
    return float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
}

float uniformConePdf(float cosThetaMax) { return 1.0 / (twoPi * (1.0 - cosThetaMax)); }

//! Heitz 2019, "A Low-Distortion Map Between Triangle and Square".
float3 sampleTriangleBarycentrics(float2 u) {
    float b0;
    float b1;
    if (u.y > u.x) {
        b0 = 0.5 * u.x;
        b1 = u.y - b0;
    } else {
        b1 = 0.5 * u.y;
        b0 = u.x - b1;
    }
    return float3(b0, b1, 1.0 - b0 - b1);
}

float powerHeuristic(float a, float b) {
    const float a2 = a * a;
    const float b2 = b * b;
    return a2 + b2 > 0.0 ? a2 / (a2 + b2) : 0.0;
}

//! Returns the chosen entry; pdf is its discrete probability.
uint sampleAlias(Ptr<AliasEntry> table, uint count, float u, out float pdf) {
    const float scaled = u * float(count);
    const uint index = min(uint(scaled), count - 1);
    const AliasEntry entry = table[index];
    const uint chosen = scaled - float(index) < entry.threshold ? index : entry.alias;
    pdf = chosen == index ? entry.pdf : table[chosen].pdf;
    return chosen;
}
