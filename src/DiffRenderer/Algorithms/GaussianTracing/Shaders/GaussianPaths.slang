import NoorRHI;
import GaussianEmission;
import GaussianScattering;
import GaussianTransport;
import Spectral.Wavelengths;

interface ITransport {
    static float4 getRadiance(Scene scene, float3 direction, inout Random random);
    static void addGradient(Scene scene, float3 direction, float4 weight, inout Random random);
}

struct Emission : ITransport {
    static float4 getRadiance(Scene scene, float3 direction, inout Random random) {
        return getEmission(scene, scene.arguments.cameraOrigin.xyz, direction);
    }

    static void addGradient(Scene scene, float3 direction, float4 weight, inout Random random) {
        addEmissionGradient(scene, scene.arguments.cameraOrigin.xyz, direction, weight);
    }
};

struct Scattering : ITransport {
    static float4 getRadiance(Scene scene, float3 direction, inout Random random) { return getScattering(scene, direction, random); }

    static void addGradient(Scene scene, float3 direction, float4 weight, inout Random random) {
        addScatteringGradient(scene, direction, weight, random);
    }
};

void render<T : ITransport>(uint2 id) {
    const Arguments arguments = rootArguments<Arguments>();
    if (id.x >= arguments.width || id.y >= arguments.height)
        return;
    const uint pixel = id.y * arguments.width + id.x;
    float3 sum = float3(0.0);
    for (uint sample = arguments.firstSample; sample < arguments.sampleEnd; ++sample) {
        Random random = Random(pixel, sample, arguments.seed);
        const float3 direction = sampleCamera(arguments, id, random);
        const Scene scene = Scene(arguments, random.next());
        sum += scene.wavelengths.toFilm(T.getRadiance(scene, direction, random));
    }
    for (uint channel = 0; channel < 3; ++channel) {
        const uint index = 3 * pixel + channel;
        const float previous = arguments.firstSample == 0 ? 0.0 : arguments.image[index];
        arguments.image[index] = previous + sum[channel] / float(arguments.samples);
    }
}

void differentiate<T : ITransport>(uint2 id) {
    const Arguments arguments = rootArguments<Arguments>();
    if (id.x >= arguments.width || id.y >= arguments.height)
        return;
    const uint pixel = id.y * arguments.width + id.x;
    const float3 weight = getWeight(arguments, pixel);
    for (uint sample = arguments.firstSample; sample < arguments.sampleEnd; ++sample) {
        Random random = Random(pixel, sample, arguments.seed);
        const float3 direction = sampleCamera(arguments, id, random);
        const Scene scene = Scene(arguments, random.next());
        T.addGradient(scene, direction, getSpectralWeight(scene.wavelengths, weight), random);
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void renderEmission(uint3 id : SV_DispatchThreadID) { render<Emission>(id.xy); }

[shader("compute")]
[numthreads(8, 8, 1)]
void renderScattering(uint3 id : SV_DispatchThreadID) { render<Scattering>(id.xy); }

[shader("compute")]
[numthreads(8, 8, 1)]
void differentiateEmission(uint3 id : SV_DispatchThreadID) { differentiate<Emission>(id.xy); }

[shader("compute")]
[numthreads(8, 8, 1)]
void differentiateScattering(uint3 id : SV_DispatchThreadID) { differentiate<Scattering>(id.xy); }
