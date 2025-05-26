const highp float PI = 3.14159265358979323846;

const float RAY_EPSYLON = 0.000001;
const float SHADOW_EPSYLON = 0.00001;

vec3 calculateBarycentric(vec3 attribs) {
    return vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
}

vec3 interpolateBarycentric(vec3 bary, vec3 p0, vec3 p1, vec3 p2) {
    return p0 * bary.x + p1 * bary.y + p2 * bary.z;
}

vec2 interpolateBarycentric(vec3 bary, vec2 p0, vec2 p1, vec2 p2) {
    return p0 * bary.x + p1 * bary.y + p2 * bary.z;
}

float D_GGX(float NdotH, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float denom = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

float G_Smith(float NdotV, float NdotL, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;

    float gv = NdotV / (NdotV * (1.0 - k) + k);
    float gl = NdotL / (NdotL * (1.0 - k) + k);
    return gv * gl;
}

vec3 Fresnel_Schlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

void createCoordinateSystem(in vec3 N, out vec3 T, out vec3 B)
{
    if(abs(N.x) > abs(N.y))
    T = vec3(N.z, 0, -N.x) / sqrt(N.x * N.x + N.z * N.z);
    else
    T = vec3(0, -N.z, N.y) / sqrt(N.y * N.y + N.z * N.z);
    B = cross(N, T);
}

vec3 sampleHemisphere(float rand1, float rand2)
{
    vec3 dir;
    float phi = 2.0 * PI * rand2;
    float cosTheta = sqrt(1.0 - rand1);
    float sinTheta = sqrt(rand1);
    dir.x = cos(phi) * sinTheta;
    dir.y = sin(phi) * sinTheta;
    dir.z = cosTheta;
    return dir;
}

vec3 sampleDirection(float rand1, float rand2, vec3 normal)
{
    vec3 tangent;
    vec3 bitangent;
    createCoordinateSystem(normal, tangent, bitangent);
    vec3 dir = sampleHemisphere(rand1, rand2);
    return dir.x * tangent + dir.y * bitangent + dir.z * normal;
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

float rand(inout uint seed)
{
    uint val = pcg(seed);
    return (float(val) * (1.0 / float(0xffffffffu)));
}