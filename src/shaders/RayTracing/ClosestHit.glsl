#version 460
#pragma shader_stage(closest)

#extension GL_EXT_ray_tracing : enable
#extension GL_GOOGLE_include_directive : enable

#include "../SharedStructs.h"
#include "../Common.glsl"

// Buffers
layout(binding = 2, set = 0) buffer Vertices { Vertex vertices[]; };
layout(binding = 3, set = 0) buffer Indices { uint indices[]; };
layout(binding = 4, set = 0) buffer Faces { Face faces[]; };
layout(binding = 5, set = 0) buffer Materials { Material materials[]; };
layout(binding = 6, set = 0) buffer PointLights { PointLight pointLights[]; };

layout(location = 0) rayPayloadInEXT PrimaryRayPayload payload;

hitAttributeEXT vec3 attribs;

layout(push_constant) uniform PushConstants {
    int frame, isPathtracing, isMoving, _pad1;
    vec3 position; int _pad2;
    vec3 direction; int _pad3;
    vec3 horizontal; int _pad4;
    vec3 vertical; int _pad5;
} pushConstants;

void main() {

    const uint primitiveIndex = gl_PrimitiveID;
    const Face face = faces[primitiveIndex];
    const Material material = materials[face.materialIndex];

    //Triangle vertices
    const uint i0 = indices[3 * primitiveIndex + 0];
    const uint i1 = indices[3 * primitiveIndex + 1];
    const uint i2 = indices[3 * primitiveIndex + 2];

    const Vertex v0 = vertices[i0];
    const Vertex v1 = vertices[i1];
    const Vertex v2 = vertices[i2];

    //Barycentric
    const vec3 bary = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
    const vec3 position = v0.position * bary.x + v1.position * bary.y + v2.position * bary.z;
    const vec3 normal = normalize(v0.normal * bary.x + v1.normal * bary.y + v2.normal * bary.z);

    payload.position = position;
    payload.normal = normal;

    vec3 ambient = 0.2 * material.albedo;
    vec3 totalDiffuse = vec3(0.0);
    vec3 totalSpecular = vec3(0.0);
    vec3 viewDir = normalize(pushConstants.position - position);

    for (int i = 0; i < pointLights.length(); ++i) {
        PointLight light = pointLights[i];

        vec3 lightVec = light.position - position;
        float distance = length(lightVec);
        vec3 lightDir = normalize(lightVec);

        //Linear Falloff
        float falloff = 1.0 / distance;
        vec3 lightIntensity = light.color * light.intensity * falloff;

        //Lambert
        float NdotL = dot(normal, lightDir);

        if(NdotL < 0.0)
            continue;

        totalDiffuse += NdotL * material.albedo * lightIntensity;

        //Blinn-Phong
        vec3 halfVec = normalize(lightDir + viewDir);
        float specAngle = max(dot(normal, halfVec), 0.0);
        float specularStrength = pow(specAngle, 32.0); //Shininess
        totalSpecular += specularStrength * material.specular * lightIntensity;
    }

    //DIR LIGHT
    const vec3 dirLightDirection = normalize(vec3(0, -1.0, 0));
    const vec3 dirLightColor = vec3(1.0, 1.0, 1.0);
    const float dirLightIntensity = 0.5;

    vec3 lightDir = normalize(dirLightDirection);
    float NdotL = dot(normal, lightDir);

    if (NdotL > 0.0) {
        vec3 lightIntensity = dirLightColor * dirLightIntensity;

        // Diffuse
        totalDiffuse += NdotL * material.albedo * lightIntensity;
    }

    payload.color = material.emission + ambient + totalDiffuse + totalSpecular;
}