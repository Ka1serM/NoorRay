const highp float PI = 3.14159265358979323846;
const float EPSILON = 0.000001;

const uint BOUNCE_TYPE_DIFFUSE     = 0;
const uint BOUNCE_TYPE_SPECULAR    = 1;
const uint BOUNCE_TYPE_TRANSMISSION = 2;

vec3 calculateBarycentric(vec3 attribs) {
    return vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
}

vec3 interpolateBarycentric(vec3 bary, vec3 p0, vec3 p1, vec3 p2) {
    return p0 * bary.x + p1 * bary.y + p2 * bary.z;
}

vec2 interpolateBarycentric(vec3 bary, vec2 p0, vec2 p1, vec2 p2) {
    return p0 * bary.x + p1 * bary.y + p2 * bary.z;
}

uint pcg(inout uint state)
{
    uint prev = state * 747796405u + 2891336453u;
    uint word = ((prev >> ((prev >> 28u) + 4u)) ^ prev) * 277803737u;
    state = prev;
    return (word >> 22u) ^ word;
}

uvec2 pcg2d(uvec2 v)
{
    v = v * 1664525u + 1013904223u;
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v = v ^ (v >> 16u);
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v = v ^ (v >> 16u);
    return v;
}

// RNG float in [0,1)
float rand(inout uint seed)
{
    uint val = pcg(seed);
    return float(val) * (1.0 / 4294967296.0);
}

vec2 concentricSampleDisk(float u1, float u2) {
    float offsetX = 2.0 * u1 - 1.0;
    float offsetY = 2.0 * u2 - 1.0;

    if (offsetX == 0.0 && offsetY == 0.0)
    return vec2(0.0);

    float r, theta;
    if (abs(offsetX) > abs(offsetY)) {
        r = offsetX;
        theta = (3.14159265 / 4.0) * (offsetY / offsetX);
    } else {
        r = offsetY;
        theta = (3.14159265 / 2.0) - (3.14159265 / 4.0) * (offsetX / offsetY);
    }

    return r * vec2(cos(theta), sin(theta));
}

vec2 roundBokeh(float u1, float u2, float edgeBias) {
    // Get uniform disk sample first
    vec2 diskSample = concentricSampleDisk(u1, u2);
    float r = length(diskSample);

    // edgeBias: 1.0 = uniform, 2.0 = moderate edge bias, 10.0 = extreme edge bias, 0.5 = center bias
    float newR = pow(r, 1.0 / edgeBias);

    // Reconstruct the sample with new radius
    if (r > 0.0)
        return diskSample * (newR / r);
    return vec2(0.0);
}