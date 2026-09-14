#version 450

//#############################################################################
//#
//#   G O L D   T O R U S
//#   real-time path tracing  +  ray tracing  +  ray marching
//#
//#   Everything in the frame is gold: the environment, the studio floor and
//#   the torus itself (polished gold uses its measured optical constants as
//#   the Schlick F0 -- metalness 1, therefore no diffuse lobe).
//#
//#   ---------------------------------------------------------------------
//#   mode 0  (full-screen triangle) -- the complete path-traced frame.
//#
//#       RAY TRACING   the primary hit and *every* secondary hit against the
//#                     torus is solved analytically: substituting the ray
//#                     into the implicit torus equation gives a quartic which
//#                     is solved in closed form (Ferrari + Cardano) and then
//#                     polished with Newton steps against the exact implicit
//#                     function.  The gold floor is intersected analytically
//#                     too.
//#
//#       RAY MARCHING  the penumbra of the key light, the ambient occlusion
//#                     and the volumetric gold aura are sphere-traced against
//#                     the exact torus distance field.  Marching is also the
//#                     fallback if the closed-form solve ever degenerates.
//#
//#       PATH TRACING  the rendering equation is estimated by Monte-Carlo
//#                     integration: GGX importance sampling of the metal
//#                     lobe, next-event estimation of the key light combined
//#                     with multiple importance sampling, Russian-roulette
//#                     path termination, SPP samples per pixel per frame.
//#
//#   mode 1  (rasterised torus mesh, depth tested) -- the same path tracer,
//#           but the primary hit is delivered by the rasteriser instead of a
//#           camera ray: exact depth-tested visibility and a crisp silhouette
//#           with identical shading.  Classic hybrid renderer --
//#           "rasterise the primary, trace the secondaries".
//#
//#   ---------------------------------------------------------------------
//#   SPP is the main quality/performance knob: raise it for a cleaner image,
//#   lower it if the frame rate drops.
//#
//#############################################################################

#define SPP              4        // path-traced samples per pixel, per frame
#define MAX_BOUNCES      4        // path length (camera ray included)
#define SHADOW_STEPS     28       // sphere-tracing budget for the penumbra
#define AURA_STEPS       72       // sphere-tracing budget for the gold aura
#define MARCH_FALLBACK   90       // sphere-tracing budget (quartic fallback)
#define FIRE_CLAMP       30.0     // per-sample firefly clamp (linear light)
#define OUTPUT_SRGB      0        // set to 1 only if the swap chain is not sRGB

const float PI  = 3.14159265359;
const float TAU = 6.28318530718;
const float INF = 1.0e4;

// ---------------------------------------------------------------- geometry --
// Must match generateTorus(3.0f, 1.0f, ...) in main.cpp: a ring torus whose
// axis is +Y and whose ring circle lies in the XZ plane, in *object* space.
const float TORUS_R = 3.0;                        // major (ring) radius
const float TORUS_r = 1.0;                        // minor (tube) radius
const float BOUND_R = TORUS_R + TORUS_r;          // bounding sphere
const float AURA_R  = TORUS_R + TORUS_r + 1.35;   // aura bounding sphere
const float FLOOR_Y = -2.6;                       // gold studio floor

// ---------------------------------------------------------------- material --
const vec3  GOLD        = vec3(1.000, 0.710, 0.290);
const float ROUGH_TORUS = 0.155;
const float ROUGH_FLOOR = 0.34;

// -------------------------------------------------------------- key light --
const vec3  SUN_DIR       = vec3(-0.719680, 0.599730, -0.349840);   // normalised
const vec3  SUN_COLOR     = vec3(1.000, 0.860, 0.580);
const float SUN_INTENSITY = 240.0;
const float SUN_COS       = 0.995396;   // cos(5.5 deg) -- disc radius (soft key light)
const float SUN_COS_SOFT  = 0.992546;   // cos(7 deg) -- antialiased rim
// A big soft studio panel: gives polished metal its characteristic streak.
const vec3  PANEL_DIR     = vec3(0.417600, 0.617200, 0.666500);     // normalised
const vec3  PANEL_COLOR   = vec3(1.000, 0.850, 0.560);
const float PANEL_INTENSITY = 5.5;
const float PANEL_SHARP   = 26.0;
const float SUN_PDF       = 1.0 / (TAU * (1.0 - SUN_COS));

// ------------------------------------------------------------------- aura --
const vec3  AURA_COLOR   = vec3(1.00, 0.46, 0.10);
const float AURA_DENSITY = 0.045;
const float AURA_FALLOFF = 4.2;

// --------------------------------------------------------------- exposure --
const float EXPOSURE = 0.95;

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 params;      // x = time (s), y = frame index, z = width, w = height
} ubo;

layout(push_constant) uniform PushConstants {
    int mode;
} pc;

layout(location = 0) in  vec3 vWorldPos;
layout(location = 1) in  vec3 vWorldNormal;
layout(location = 0) out vec4 outColor;

//=============================================================================
//  Random numbers.  Module-scope variables in GLSL are per-invocation, so a
//  global generator behaves exactly like a local one.
//=============================================================================
uint rngState;

uint pcg(uint v) {
    v = v * 747796405u + 2891336453u;
    v = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
    return (v >> 22u) ^ v;
}

void rngSeed(uint s) {
    rngState = pcg(s) | 1u;
}

uint rngU() {
    rngState = pcg(rngState);
    return rngState;
}

float rngF() {
    return float(rngU() >> 8) * (1.0 / 16777216.0);
}

vec2 rngF2() {
    return vec2(rngF(), rngF());
}

//=============================================================================
//  Hash / value noise -- gold-leaf marbling and brushed-metal roughness
//=============================================================================
float hash13(vec3 p3) {
    p3 = fract(p3 * 0.1031);
    p3 = p3 + dot(p3, p3.zyx + 31.32);
    return fract((p3.x + p3.y) * p3.z);
}

float vnoise(vec3 x) {
    vec3 i = floor(x);
    vec3 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    float n000 = hash13(i + vec3(0.0, 0.0, 0.0));
    float n100 = hash13(i + vec3(1.0, 0.0, 0.0));
    float n010 = hash13(i + vec3(0.0, 1.0, 0.0));
    float n110 = hash13(i + vec3(1.0, 1.0, 0.0));
    float n001 = hash13(i + vec3(0.0, 0.0, 1.0));
    float n101 = hash13(i + vec3(1.0, 0.0, 1.0));
    float n011 = hash13(i + vec3(0.0, 1.0, 1.0));
    float n111 = hash13(i + vec3(1.0, 1.0, 1.0));
    return mix(mix(mix(n000, n100, f.x), mix(n010, n110, f.x), f.y),
               mix(mix(n001, n101, f.x), mix(n011, n111, f.x), f.y), f.z);
}

float fbm2(vec3 p) {
    return 0.62 * vnoise(p) + 0.38 * vnoise(p * 2.17 + vec3(11.3, 7.1, 3.7));
}

//=============================================================================
//  Object space.  The model matrix is a pure rotation, so its inverse is its
//  transpose and no separate normal matrix is needed.
//=============================================================================
mat3 objRot() {
    return transpose(mat3(ubo.model));
}

// The model matrix carries a rigid rotation plus a translation (the torus
// floats above the studio floor), so the world<->object maps are affine.
vec3 torusCentre() {
    vec4 c = ubo.model[3];
    return c.xyz;
}

vec3 toObj(vec3 w) {
    return objRot() * (w - torusCentre());
}

vec3 toWorld(vec3 o) {
    return mat3(ubo.model) * o + torusCentre();
}

// Direction variants: rotations yes, translations no.
vec3 toObjDir(vec3 d) {
    return objRot() * d;
}

vec3 toWorldDir(vec3 d) {
    return mat3(ubo.model) * d;
}

//=============================================================================
//  RAY MARCHING -- exact signed distance field of a ring torus and everything
//  derived from it: penumbra, occlusion, volumetric aura, fallback tracing.
//=============================================================================
float sdTorus(vec3 p) {
    vec2 q = vec2(length(p.xz) - TORUS_R, p.y);
    return length(q) - TORUS_r;
}

float sdTorusW(vec3 pWorld) {
    return sdTorus(toObj(pWorld));
}

vec3 torusNormal(vec3 p) {
    float q = max(length(p.xz), 1.0e-6);
    vec3 ring = vec3(p.x * TORUS_R / q, 0.0, p.z * TORUS_R / q);
    return normalize(p - ring);
}

// iq's soft shadow: sphere-trace towards the light while tracking penumbra.
float softShadow(vec3 ro, vec3 rd, float tmin, float tmax, float k) {
    float res = 1.0;
    float t = tmin;
    for (int i = 0; i < SHADOW_STEPS; ++i) {
        vec3 p = ro + rd * t;
        if (dot(p, p) > 900.0) break;               // left the neighbourhood
        float h = sdTorusW(p);
        if (h < 0.0015) return 0.0;
        res = min(res, k * h / t);
        t = t + clamp(h, 0.02, 0.6);
        if (t > tmax) break;
    }
    return clamp(res, 0.0, 1.0);
}

float ambientOcclusion(vec3 p, vec3 n) {
    float occ = 0.0;
    float sca = 1.0;
    for (int i = 0; i < 5; ++i) {
        float h = 0.03 + 0.16 * float(i);
        float d = sdTorusW(p + n * h);
        occ = occ + (h - d) * sca;
        sca = sca * 0.82;
    }
    return clamp(1.0 - 1.6 * occ, 0.0, 1.0);
}

// Volumetric gold aura: sphere-trace the shell around the torus and integrate
// exp(-falloff * distance) along the ray.  It gives the metal a warm bloom and
// welds the torus into the background instead of cutting it out of it.
vec3 goldAura(vec3 ro, vec3 rd, float tmax) {
    float b = dot(ro, rd);
    float c = dot(ro, ro) - AURA_R * AURA_R;
    float disc = b * b - c;
    if (disc < 0.0) return vec3(0.0);
    disc = sqrt(disc);
    float t = max(-b - disc, 0.0);
    float tEnd = min(-b + disc, tmax);
    if (t >= tEnd) return vec3(0.0);

    float acc = 0.0;
    for (int i = 0; i < AURA_STEPS; ++i) {
        vec3 p = ro + rd * t;
        float d = sdTorusW(p);
        if (d < 0.0) break;                         // entered the metal
        float dt = max(d * 0.55, 0.015);
        if (t + dt > tEnd) dt = max(tEnd - t, 0.0);
        acc = acc + exp(-d * AURA_FALLOFF) * dt;
        t = t + dt;
        if (t >= tEnd || acc > 4.0) break;
    }
    return AURA_COLOR * (AURA_DENSITY * 4.0 * acc);
}

//=============================================================================
//  RAY TRACING -- analytic ray / torus intersection.
//
//  Implicit form (axis = Y):  ( |p|^2 + R^2 - r^2 )^2 - 4 R^2 (x^2 + z^2) = 0
//  Substituting p = ro + t rd gives the quartic
//      t^4 + c3 t^3 + c2 t^2 + c1 t + c0 = 0
//  solved in closed form below (depress -> resolvent cubic -> two quadratics)
//  and then polished with Newton steps on the implicit function, so float32
//  round-off cannot leak into the shading.
//=============================================================================
struct Roots {
    int   n;
    float r[4];
};

Roots noRoots() {
    Roots res;
    res.n = 0;
    res.r[0] = 0.0;
    res.r[1] = 0.0;
    res.r[2] = 0.0;
    res.r[3] = 0.0;
    return res;
}

float cbrtR(float x) {
    return sign(x) * pow(abs(x), 1.0 / 3.0);
}

// x^3 + b x^2 + c x + d = 0
Roots solveCubic(float b, float c, float d) {
    Roots res = noRoots();

    float p = c - b * b / 3.0;
    float q = 2.0 * b * b * b / 27.0 - b * c / 3.0 + d;
    float shift = -b / 3.0;
    float disc = q * q * 0.25 + p * p * p / 27.0;

    if (disc > 0.0) {
        float s = sqrt(disc);
        res.r[0] = cbrtR(-q * 0.5 + s) + cbrtR(-q * 0.5 - s) + shift;
        res.n = 1;
    } else if (p > -1.0e-8) {
        res.r[0] = cbrtR(-q) + shift;
        res.n = 1;
    } else {
        float m = 2.0 * sqrt(-p / 3.0);
        float t = clamp(3.0 * q / (p * m), -1.0, 1.0);
        float phi = acos(t) / 3.0;
        res.r[0] = m * cos(phi) + shift;
        res.r[1] = m * cos(phi - TAU / 3.0) + shift;
        res.r[2] = m * cos(phi - 2.0 * TAU / 3.0) + shift;
        res.n = 3;
    }
    return res;
}

// x^4 + b x^3 + c x^2 + d x + e = 0     (Ferrari)
Roots solveQuartic(float b, float c, float d, float e) {
    Roots res = noRoots();

    float p = c - 3.0 * b * b / 8.0;
    float q = b * b * b / 8.0 - b * c / 2.0 + d;
    float r = -3.0 * b * b * b * b / 256.0 + b * b * c / 16.0 - b * d / 4.0 + e;
    float shift = -b / 4.0;

    if (abs(q) < 1.0e-7) {                          // biquadratic
        float disc = p * p - 4.0 * r;
        if (disc < 0.0) return res;
        float s = sqrt(disc);
        float y0 = (-p + s) * 0.5;
        float y1 = (-p - s) * 0.5;
        if (y0 >= 0.0) {
            float z = sqrt(y0);
            res.r[res.n] =  z + shift; res.n = res.n + 1;
            res.r[res.n] = -z + shift; res.n = res.n + 1;
        }
        if (y1 >= 0.0) {
            float z = sqrt(y1);
            res.r[res.n] =  z + shift; res.n = res.n + 1;
            res.r[res.n] = -z + shift; res.n = res.n + 1;
        }
        return res;
    }

    // resolvent:  m^3 - (p/2) m^2 - r m + (p r / 2 - q^2 / 8) = 0
    Roots cr = solveCubic(-p * 0.5, -r, p * r * 0.5 - q * q / 8.0);
    if (cr.n == 0) return res;
    float m = cr.r[0];
    for (int i = 1; i < cr.n; ++i) m = max(m, cr.r[i]);

    float twoMP = 2.0 * m - p;
    if (twoMP <= 1.0e-9) return res;
    float sq = sqrt(twoMP);
    float s  = q / (2.0 * sq);

    float d1 = sq * sq - 4.0 * (m + s);             // x^2 - sq x + (m + s) = 0
    if (d1 >= 0.0) {
        float z = sqrt(d1);
        res.r[res.n] = ( sq - z) * 0.5 + shift; res.n = res.n + 1;
        res.r[res.n] = ( sq + z) * 0.5 + shift; res.n = res.n + 1;
    }
    float d2 = sq * sq - 4.0 * (m - s);             // x^2 + sq x + (m - s) = 0
    if (d2 >= 0.0) {
        float z = sqrt(d2);
        res.r[res.n] = (-sq - z) * 0.5 + shift; res.n = res.n + 1;
        res.r[res.n] = (-sq + z) * 0.5 + shift; res.n = res.n + 1;
    }
    return res;
}

struct Hit {
    float t;        // < 0 on a miss
    vec3  p;        // world space
    vec3  n;        // world space, unit
    int   mat;      // 0 = none, 1 = gold torus, 2 = gold floor
};

Hit missHit() {
    Hit h;
    h.t = -1.0;
    h.p = vec3(0.0);
    h.n = vec3(0.0, 1.0, 0.0);
    h.mat = 0;
    return h;
}

// Sphere-traced fallback: only reached if the closed-form solve degenerates.
Hit traceTorusMarch(vec3 ro, vec3 rd, float tmin, float tmax) {
    Hit h = missHit();
    float t = tmin;
    for (int i = 0; i < MARCH_FALLBACK; ++i) {
        vec3 p = ro + rd * t;
        float d = sdTorus(p);
        if (d < 0.0004 * max(1.0, abs(t))) {
            h.t = t;
            h.p = toWorld(p);
            h.n = toWorldDir(torusNormal(p));
            h.mat = 1;
            return h;
        }
        t = t + d;
        if (t > tmax) break;
    }
    return h;
}

// Three clamped Newton steps on the implicit torus function.
float polishRoot(vec3 ro, vec3 rd, float R2, float r2, float t) {
    for (int i = 0; i < 3; ++i) {
        vec3 p = ro + rd * t;
        float f = dot(p, p) + R2 - r2;
        float F = f * f - 4.0 * R2 * (p.x * p.x + p.z * p.z);
        vec3  G = 4.0 * f * p - 8.0 * R2 * vec3(p.x, 0.0, p.z);
        float dF = dot(G, rd);
        if (abs(dF) < 1.0e-6) break;
        t = t - clamp(F / dF, -0.25, 0.25);
    }
    return t;
}

// traceTorusMarch runs in the shifted object-space parameterisation, so its t
// has to be moved back -- and a genuine miss has to stay a miss (t' may legally
// be negative, so `mat` is the hit flag, not the sign of t).
Hit marchFallback(vec3 roObj, vec3 rdObj, float t0, float t1, float tc) {
    Hit fb = traceTorusMarch(roObj, rdObj, t0, t1);
    if (fb.mat == 1) fb.t = fb.t + tc;
    else             fb.t = -1.0;
    return fb;
}

Hit traceTorus(vec3 roW, vec3 rdW, float tmin, float tmax) {
    Hit h = missHit();
    vec3 ro = toObj(roW);
    vec3 rd = toObjDir(rdW);

    // Re-parameterise so the ray origin is its closest point to the torus
    // centre.  That makes dot(ro, rd) == 0, which kills the cubic term of the
    // quartic and shrinks every coefficient by orders of magnitude (|ro| drops
    // from ~15 to the impact parameter, <= 4).  Without this shift a float32
    // Ferrari solve loses the roots to catastrophic cancellation.
    float tc = -dot(ro, rd);
    ro = ro + rd * tc;

    float b2 = dot(ro, ro);
    float chord = BOUND_R * BOUND_R - b2;             // bounding-sphere reject
    if (chord < -0.01) return h;
    chord = sqrt(max(chord, 0.0));
    // Pad the interval: b2 is a difference of two large numbers, so the sphere
    // boundary is only accurate to a few 1e-6 and a genuine entry root sitting
    // exactly on it would otherwise be rejected.  Roots of the quartic lie on
    // the torus by construction, so widening the window cannot admit a ghost.
    float pad = 1.0e-3;
    float t0 = max(-chord - pad, tmin - tc);
    float t1 = min( chord + pad, tmax - tc);
    if (t0 > t1) return h;

    float R2 = TORUS_R * TORUS_R;
    float r2 = TORUS_r * TORUS_r;
    float A  = b2 + R2 - r2;                          // == dot(ro,ro)+R^2-r^2
    float d2 = rd.x * rd.x + rd.z * rd.z;
    float d1 = ro.x * rd.x + ro.z * rd.z;
    float d0 = ro.x * ro.x + ro.z * ro.z;

    // t^4 + (2A - 4R^2 d2) t^2 - 8R^2 d1 t + (A^2 - 4R^2 d0) = 0
    Roots rr = solveQuartic(0.0,
                            2.0 * A - 4.0 * R2 * d2,
                            -8.0 * R2 * d1,
                            A * A - 4.0 * R2 * d0);

    // The closed-form roots carry a few 1e-3 of float32 error (the resolvent
    // cubic cancels), so polish *every* root against the exact implicit
    // function before testing it against the interval -- otherwise a good root
    // can be gated out by its own error.  The Newton step is clamped: at a
    // grazing hit dF/dt -> 0 and an unclamped step would launch the root away.
    for (int i = 0; i < rr.n; ++i) {
        rr.r[i] = polishRoot(ro, rd, R2, r2, rr.r[i]);
    }

    float best = t1 + 1.0;
    for (int i = 0; i < rr.n; ++i) {
        float t = rr.r[i];
        if (t > t0 && t < best) best = t;
    }
    if (best > t1) return marchFallback(ro, rd, t0, t1, tc);

    float t = best;

    // Trust but verify: if the polished root drifted off the surface, sphere
    // trace the same interval instead.
    if (t < t0 - 1.0e-4 || t > t1 + 1.0e-4 || abs(sdTorus(ro + rd * t)) > 2.0e-3) {
        return marchFallback(ro, rd, t0, t1, tc);
    }

    vec3 po = ro + rd * t;
    h.t   = t + tc;
    h.p   = toWorld(po);
    h.n   = normalize(toWorldDir(torusNormal(po)));
    h.mat = 1;
    return h;
}

Hit traceFloor(vec3 ro, vec3 rd, float tmin, float tmax) {
    Hit h = missHit();
    if (abs(rd.y) < 1.0e-6) return h;
    float t = (FLOOR_Y - ro.y) / rd.y;
    if (t < tmin || t > tmax) return h;
    h.t   = t;
    h.p   = ro + rd * t;
    h.n   = vec3(0.0, 1.0, 0.0);
    h.mat = 2;
    return h;
}

Hit intersectScene(vec3 ro, vec3 rd) {
    Hit a = traceTorus(ro, rd, 0.0008, INF);
    Hit b = traceFloor(ro, rd, 0.0008, INF);
    if (a.t < 0.0) return b;
    if (b.t < 0.0) return a;
    return (a.t < b.t) ? a : b;
}

//=============================================================================
//  The gold environment.  This *is* the background: a graded gold studio
//  backdrop marbled like gold leaf, plus a warm key light with a bloom.
//=============================================================================
vec3 skyBase(vec3 d) {
    float hgt = d.y * 0.5 + 0.5;

    vec3 zenith  = vec3(0.30, 0.20, 0.085);  // deep antique gold
    vec3 horizon = vec3(0.92, 0.48, 0.11);   // glowing gold band
    vec3 nadir   = vec3(0.050, 0.024, 0.008);// near-black bronze

    vec3 c = mix(horizon, zenith, smoothstep(0.50, 1.00, hgt));
    c = mix(nadir, c, smoothstep(0.00, 0.50, hgt));

    // gold-leaf marbling
    float m = fbm2(d * 2.4);
    c = c * (0.70 + 0.60 * m);

    // broad key glow + tight bloom around the sun
    float sd = max(dot(d, SUN_DIR), 0.0);
    c = c + SUN_COLOR * 0.30 * pow(sd, 6.0);
    c = c + SUN_COLOR * 2.00 * pow(sd, 140.0);

    // studio panel: the bright streak that makes polished metal read as metal
    float pd = max(dot(d, PANEL_DIR), 0.0);
    c = c + PANEL_COLOR * PANEL_INTENSITY * pow(pd, PANEL_SHARP);

    // warm bounce fill from the opposite side
    float fd = max(dot(d, -SUN_DIR), 0.0);
    c = c + vec3(0.28, 0.15, 0.055) * 0.35 * pow(fd, 3.0);

    // horizon haze
    float hz = max(1.0 - abs(d.y), 0.0);
    c = c + vec3(0.90, 0.52, 0.18) * 0.30 * pow(hz, 10.0);

    return c;
}

vec3 sunDisc(vec3 d) {
    float c = dot(d, SUN_DIR);
    float e = smoothstep(SUN_COS_SOFT, SUN_COS, c);
    return SUN_COLOR * SUN_INTENSITY * e;
}

vec3 sky(vec3 d) {
    return skyBase(d) + sunDisc(d);
}

//=============================================================================
//  Microfacet BRDF -- Cook-Torrance, metalness 1, so there is no diffuse lobe
//=============================================================================
float ggxD(float ndh, float a) {
    float a2 = a * a;
    float d = ndh * ndh * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

float ggxG1(float ndx, float a) {
    float a2 = a * a;
    return 2.0 * ndx / (ndx + sqrt(a2 + (1.0 - a2) * ndx * ndx));
}

vec3 fresnelF0(vec3 f0, float voh) {
    float f = pow(clamp(1.0 - voh, 0.0, 1.0), 5.0);
    return f0 + (vec3(1.0) - f0) * f;
}

struct ONB {
    vec3 t;
    vec3 b;
};

// branchless orthonormal basis (Drobot / Hoffman)
ONB makeONB(vec3 n) {
    ONB o;
    float s = (n.z >= 0.0) ? 1.0 : -1.0;
    float a = -1.0 / (s + n.z);
    float bb = n.x * n.y * a;
    o.t = vec3(1.0 + s * n.x * n.x * a, s * bb, -s * n.x);
    o.b = vec3(bb, s + n.y * n.y * a, -n.y);
    return o;
}

vec3 sampleGGX(vec3 n, float rough, vec2 u) {
    float a = max(rough * rough, 1.0e-4);
    float phi = TAU * u.x;
    float cosT = sqrt((1.0 - u.y) / (1.0 + (a * a - 1.0) * u.y));
    float sinT = sqrt(max(0.0, 1.0 - cosT * cosT));
    ONB o = makeONB(n);
    return normalize(o.t * (sinT * cos(phi)) + o.b * (sinT * sin(phi)) + n * cosT);
}

// pdf of the *reflected direction* produced by sampleGGX
float pdfGGX(vec3 n, vec3 h, float rough, float voh) {
    float a = max(rough * rough, 1.0e-4);
    float ndh = max(dot(n, h), 0.0);
    return ggxD(ndh, a) * ndh / max(4.0 * voh, 1.0e-5);
}

vec3 sampleSun(vec2 u) {
    float cosT = mix(SUN_COS, 1.0, u.x);
    float sinT = sqrt(max(0.0, 1.0 - cosT * cosT));
    float phi = TAU * u.y;
    ONB o = makeONB(SUN_DIR);
    return normalize(o.t * (sinT * cos(phi)) + o.b * (sinT * sin(phi)) + SUN_DIR * cosT);
}

// Brushed gold: fine seamless streaks around the tube (integer frequencies
// keep the atan() wrap continuous, so there is no seam on the metal).
float torusRoughness(vec3 pWorld) {
    vec3 p = toObj(pWorld);
    float v = atan(p.y, length(p.xz) - TORUS_R);
    float streak = 0.5 + 0.5 * sin(v * 40.0 + 1.7 * sin(v * 13.0));
    return ROUGH_TORUS * (0.86 + 0.30 * streak);
}

float floorRoughness(vec3 p) {
    float streak = fbm2(vec3(p.x * 0.55, 0.0, p.z * 0.55));
    return ROUGH_FLOOR * (0.80 + 0.45 * streak);
}

struct Mat {
    float metal;    // 1 = pure metal (the torus), <1 adds a diffuse satin base
    float rough;
    vec3  f0;
    vec3  albedo;
};

// The torus is polished gold: metalness 1, no diffuse lobe.  The floor is
// satin gold paint -- 80% metal, 20% diffuse -- which is what lets the
// sphere-traced penumbra of the key light read as a real shadow.
Mat materialAt(Hit h) {
    Mat m;
    if (h.mat == 1) {
        m.metal  = 1.0;
        m.rough  = torusRoughness(h.p);
        m.f0     = GOLD;
        m.albedo = vec3(0.0);
    } else {
        m.metal  = 0.8;
        m.rough  = floorRoughness(h.p);
        m.f0     = GOLD * 0.9 + vec3(0.02);
        m.albedo = vec3(0.40, 0.24, 0.09);
    }
    return m;
}

vec3 cosineHemisphere(vec3 n, vec2 u) {
    float r = sqrt(u.x);
    float phi = TAU * u.y;
    ONB o = makeONB(n);
    return normalize(o.t * (r * cos(phi)) + o.b * (r * sin(phi)) + n * sqrt(max(0.0, 1.0 - u.x)));
}


//=============================================================================
//  PATH TRACING
//
//  Walks a path from a known first hit.  At every vertex:
//     - next event estimation towards the key light with a sphere-traced
//       penumbra, MIS-combined with the BRDF sample (power heuristic),
//     - one GGX importance-sampled bounce whose next vertex is found by
//       analytic ray tracing,
//     - Russian roulette: unbiased, and it keeps short paths cheap.
//=============================================================================
vec3 pathTrace(vec3 rd, Hit hit) {
    vec3 L = vec3(0.0);          // accumulated radiance
    vec3 T = vec3(1.0);          // path throughput

    for (int bounce = 0; bounce < MAX_BOUNCES; ++bounce) {
        vec3 p = hit.p;
        vec3 n = hit.n;
        vec3 v = -rd;

        if (dot(n, v) < 0.0) n = -n;                // always face the incoming ray

        // the floor dissolves into the backdrop far away (no hard horizon)
        float fade = 1.0;
        if (hit.mat == 2) fade = 1.0 - smoothstep(13.0, 34.0, length(p.xz));
        if (fade < 1.0) {
            L = L + T * (sky(rd) * (1.0 - fade));
            T = T * fade;
            if (fade < 0.002) break;
        }

        Mat m = materialAt(hit);
        float rough = m.rough;
        float a = rough * rough;
        float ndv = max(dot(n, v), 1.0e-4);

        float ao = 1.0;
        if (hit.mat == 1 && bounce == 0) ao = mix(1.0, ambientOcclusion(p, n), 0.45);
        if (hit.mat == 2 && bounce == 0) ao = mix(1.0, ambientOcclusion(p, n), 0.55);

        // ---- next event estimation: key light, MIS'd against the lobe ------
        vec3  ls  = sampleSun(rngF2());
        float ndl = dot(n, ls);
        if (ndl > 0.0) {
            vec3  hh  = normalize(ls + v);
            float ndh = max(dot(n, hh), 0.0);
            float voh = max(dot(v, hh), 1.0e-4);
            vec3  F   = fresnelF0(m.f0, voh);
            float G   = ggxG1(ndv, a) * ggxG1(ndl, a);
            vec3  f   = ggxD(ndh, a) * F * G / (4.0 * ndv * ndl);

            float pBrdf  = pdfGGX(n, hh, rough, voh);
            float pLight = SUN_PDF;
            float mis = pLight * pLight / (pLight * pLight + pBrdf * pBrdf);

            vec3  origin = p + n * 0.004;
            float shadow = softShadow(origin, ls, 0.02, 26.0, 9.0);
            vec3  sunRad = SUN_COLOR * SUN_INTENSITY;
            L = L + T * (f * (ndl * mis * ao / pLight)) * sunRad * shadow;

            // diffuse satin base, MIS'd against the same light sample
            if (m.metal < 1.0) {
                float pDiff = ndl / PI;
                float misD = pLight * pLight / (pLight * pLight + pDiff * pDiff);
                vec3 fdiff = m.albedo * (1.0 - m.metal) / PI;
                L = L + T * (fdiff * (ndl * misD / pLight)) * sunRad * shadow;
            }
        }

        // ---- one importance-sampled bounce ---------------------------------
        // randomly pick a lobe: specular metal, or (for satin surfaces) diffuse
        float xi = rngF();
        if (xi < m.metal) {
            vec3  h    = sampleGGX(n, rough, rngF2());
            float ndh  = max(dot(n, h), 0.0);
            float voh  = max(dot(v, h), 1.0e-4);
            vec3  l    = reflect(rd, h);
            float ndl2 = dot(n, l);
            if (ndl2 <= 0.0) break;               // zero-weight lobe: path ends

            vec3  F = fresnelF0(m.f0, voh);
            float G = ggxG1(ndv, a) * ggxG1(ndl2, a);
            vec3  w = F * (G * voh / max(ndv * ndh, 1.0e-4)) / m.metal;

            Hit next = intersectScene(p + n * 0.004, l);
            if (next.t < 0.0) {
                // escaped: the gold environment closes the path.  The sun disc
                // is MIS-weighted against the explicit light sample above.
                float pBrdf  = pdfGGX(n, h, rough, voh);
                float pLight = SUN_PDF;
                float mis = pBrdf * pBrdf / (pBrdf * pBrdf + pLight * pLight);
                vec3 env = skyBase(l) + sunDisc(l) * mis;
                L = L + T * w * env;
                break;
            }

            T = T * w;
            rd = l;
            hit = next;
        } else {
            vec3 l = cosineHemisphere(n, rngF2());
            float ndl2 = dot(n, l);
            if (ndl2 <= 0.0) break;
            // brdf*ndl/pdf = albedo*(1-metal); divided by the branch
            // probability (1-metal) leaves the plain albedo.
            T = T * m.albedo;
            rd = l;
            hit = intersectScene(p + n * 0.004, l);
            if (hit.t < 0.0) { L = L + T * sky(rd); break; }
        }

        // ---- Russian roulette ----------------------------------------------
        if (bounce >= 1) {
            float q = clamp(max(T.x, max(T.y, T.z)), 0.08, 0.95);
            if (rngF() > q) break;
            T = T / q;
        }
    }

    return L;
}

//=============================================================================
//  Camera + presentation
//=============================================================================
vec3 cameraOrigin() {
    mat4 iv = inverse(ubo.view);
    vec4 c = iv[3];
    return c.xyz;
}

struct Ray {
    vec3 ro;
    vec3 rd;
};

// Unprojects through the very same proj*view the rasteriser uses, so the
// traced image and the mesh pass can never drift apart.
Ray cameraRay(vec2 fragCoord, vec2 jitter) {
    Ray r;
    vec4 prm = ubo.params;
    vec2 res = prm.zw;
    vec2 ndc = vec2(((fragCoord.x + jitter.x) / res.x) * 2.0 - 1.0,
                    ((fragCoord.y + jitter.y) / res.y) * 2.0 - 1.0);
    mat4 invVP = inverse(ubo.proj * ubo.view);
    vec4 pn = invVP * vec4(ndc.x, ndc.y, 0.0, 1.0);
    vec4 pf = invVP * vec4(ndc.x, ndc.y, 1.0, 1.0);
    vec3 p0 = pn.xyz / pn.w;
    vec3 p1 = pf.xyz / pf.w;
    r.ro = p0;
    r.rd = normalize(p1 - p0);
    return r;
}

vec3 aces(vec3 x) {
    const float A = 2.51;
    const float B = 0.03;
    const float C = 2.43;
    const float D = 0.59;
    const float E = 0.14;
    return clamp((x * (A * x + B)) / (x * (C * x + D) + E), 0.0, 1.0);
}

vec3 toDisplay(vec3 linear) {
#if OUTPUT_SRGB
    linear = max(linear, vec3(0.0));
    return mix(linear * 12.92,
               1.055 * pow(linear, vec3(1.0 / 2.4)) - 0.055,
               step(vec3(0.0031308), linear));
#else
    return linear;
#endif
}

void main() {
    vec4 prm = ubo.params;
    vec2 fragCoord = gl_FragCoord.xy;
    uint frame = uint(prm.y);
    uint seedBase = uint(fragCoord.x) * 1973u + uint(fragCoord.y) * 9277u + frame * 26699u + 1u;

    vec3 col = vec3(0.0);

    if (pc.mode == 0) {
        //---------------- full path-traced frame: trace the primary ray ------
        for (int s = 0; s < SPP; ++s) {
            rngSeed(seedBase + uint(s) * 37u);
            vec2 jitter = rngF2() - vec2(0.5);
            Ray r = cameraRay(fragCoord, jitter);
            Hit hit = intersectScene(r.ro, r.rd);

            vec3 radi;
            if (hit.t < 0.0) radi = sky(r.rd);
            else             radi = pathTrace(r.rd, hit);
            radi = radi + goldAura(r.ro, r.rd, (hit.t < 0.0) ? INF : hit.t);

            col = col + min(radi, vec3(FIRE_CLAMP));
        }
        col = col / float(SPP);
    } else {
        //--------- rasterised primary hit, traced secondaries ---------------
        vec3 p = vWorldPos;
        vec3 n = normalize(vWorldNormal);
        vec3 ro = cameraOrigin();
        vec3 rd = normalize(p - ro);

        for (int s = 0; s < SPP; ++s) {
            rngSeed(seedBase + uint(s) * 37u);

            Hit hit;
            hit.t = length(p - ro);
            hit.p = p;
            hit.n = n;
            hit.mat = 1;

            vec3 radi = pathTrace(rd, hit);
            radi = radi + goldAura(ro, rd, hit.t);
            col = col + min(radi, vec3(FIRE_CLAMP));
        }
        col = col / float(SPP);
    }

    // vignette
    vec2 res = prm.zw;
    vec2 uv = fragCoord / res;
    float vig = 1.0 - 0.30 * pow(length(uv - vec2(0.5)) * 1.44, 2.2);

    col = aces(col * EXPOSURE) * vig;

    // dither: kills 8-bit banding in the gold gradient
    rngSeed(seedBase * 3u + 7919u);
    col = col + (rngF() - 0.5) * (1.0 / 255.0);

    outColor = vec4(toDisplay(clamp(col, 0.0, 1.0)), 1.0);
}
