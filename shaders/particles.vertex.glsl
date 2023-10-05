#version 460 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec4 aColor;
layout (location = 2) in vec3 offset;

out vec4 fragColor;

void main()
{
  vec3 pos = aPos + offset;
  gl_Position = vec4(pos, 1.0);
  fragColor = aColor;
}
