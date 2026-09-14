#version 450

// =====================================================================
//  vert.vert -- one shader module, two draw modes (selected by a push
//  constant, see TorusApp::createCommandBuffers in main.cpp)
//
//    mode 0 : full-screen triangle.  Vertex attributes are ignored; the
//             fragment shader path traces the whole frame.
//    mode 1 : the tessellated torus mesh.  Used as a hardware depth
//             pre-pass for the primary hit: it hands the fragment shader
//             the exact world-space position/normal of the surface, every
//             secondary ray is still traced/marched in the fragment stage.
//
//  NOTE: the uniform block declaration below must stay byte-identical to
//  the one in frag.frag (and to `struct UniformBufferObject` in main.cpp).
// =====================================================================

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 params;      // x = time (s), y = frame index, z = width, w = height
} ubo;

layout(push_constant) uniform PushConstants {
    int mode;
} pc;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

layout(location = 0) out vec3 vWorldPos;
layout(location = 1) out vec3 vWorldNormal;

void main() {
    if (pc.mode == 0) {
        // Full-screen triangle: (0,0), (2,0), (0,2) in [0,2] clip space.
        vec2 q = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
        vWorldPos    = vec3(0.0);
        vWorldNormal = vec3(0.0);
        gl_Position  = vec4(q * 2.0 - 1.0, 0.0, 1.0);
    } else {
        vec4 world    = ubo.model * vec4(inPosition, 1.0);
        vWorldPos     = world.xyz;
        vWorldNormal  = mat3(ubo.model) * inNormal;   // model is a pure rotation
        gl_Position   = ubo.proj * ubo.view * world;
    }
}
