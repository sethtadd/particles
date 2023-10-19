#version 460 core

layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

in vec4 vertexColor[];
out vec4 geomColor;
out vec2 TexCoords;

uniform mat4 view;
uniform mat4 projection;

uniform float particleRadius;

void main() {
    geomColor = vertexColor[0];

    vec4 position = gl_in[0].gl_Position;
    float size = particleRadius * 2.0f;  // You can adjust the size of the quad

    vec4 offset1 = vec4(-size / 2, -size / 2, 0.0, 0.0);
    vec4 offset2 = vec4(size / 2, -size / 2, 0.0, 0.0);
    vec4 offset3 = vec4(-size / 2, size / 2, 0.0, 0.0);
    vec4 offset4 = vec4(size / 2, size / 2, 0.0, 0.0);

    TexCoords = vec2(0.0, 0.0);
    gl_Position = projection * (position + offset1);
    EmitVertex();

    TexCoords = vec2(1.0, 0.0);
    gl_Position = projection * (position + offset2);
    EmitVertex();

    TexCoords = vec2(0.0, 1.0);
    gl_Position = projection * (position + offset3);
    EmitVertex();

    TexCoords = vec2(1.0, 1.0);
    gl_Position = projection * (position + offset4);
    EmitVertex();

    EndPrimitive();
}
