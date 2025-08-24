#ifndef _COMMON_GLSL_ //include guard to prevent multiple inclusions
#define _COMMON_GLSL_

const float PI = 3.14159265358979323846;
const float EPSILON = 0.0001;
const float INF = 1.0e30;

// --- Bounce Types ---
const uint BOUNCE_TYPE_DIFFUSE = 0;
const uint BOUNCE_TYPE_SPECULAR = 1;
const uint BOUNCE_TYPE_TRANSMISSION = 2;

// --- Barycentric Helpers ---
vec3 calculateBarycentric(vec3 attribs) {
    return vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
}

vec3 interpolateBarycentric(vec3 bary, vec3 p0, vec3 p1, vec3 p2) {
    return p0 * bary.x + p1 * bary.y + p2 * bary.z;
}

vec2 interpolateBarycentric(vec3 bary, vec2 p0, vec2 p1, vec2 p2) {
    return p0 * bary.x + p1 * bary.y + p2 * bary.z;
}

// --- PCG Random Number Generation ---
uint pcg(inout uint state) {
    uint prev = state * 747796405u + 2891336453u;
    uint word = ((prev >> ((prev >> 28u) + 4u)) ^ prev) * 277803737u;
    state = prev;
    return (word >> 22u) ^ word;
}

uvec2 pcg2d(uvec2 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v = v ^ (v >> 16u);
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v = v ^ (v >> 16u);
    return v;
}

float rand(inout uint seed) {
    uint val = pcg(seed);
    return float(val) * (1.0 / 4294967296.0);
}

#endif