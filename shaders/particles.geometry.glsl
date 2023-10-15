#version 460 core

layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

in vec4 vertexColor[];
out vec4 geomColor;

out vec2 TexCoords;

void main() {
    vec4 position = gl_in[0].gl_Position;
    float size = 0.01;  // You can adjust the size of the quad

    geomColor = vertexColor[0];

    vec4 offset1 = vec4(-size / 2, -size / 2, 0.0, 0.0);
    vec4 offset2 = vec4(size / 2, -size / 2, 0.0, 0.0);
    vec4 offset3 = vec4(-size / 2, size / 2, 0.0, 0.0);
    vec4 offset4 = vec4(size / 2, size / 2, 0.0, 0.0);

    TexCoords = vec2(0.0, 0.0);
    gl_Position = position + offset1;
    EmitVertex();

    TexCoords = vec2(1.0, 0.0);
    gl_Position = position + offset2;
    EmitVertex();

    TexCoords = vec2(0.0, 1.0);
    gl_Position = position + offset3;
    EmitVertex();

    TexCoords = vec2(1.0, 1.0);
    gl_Position = position + offset4;
    EmitVertex();

    EndPrimitive();
}
