#version 460 core

layout (location = 0) in vec4 aColor;
layout (location = 1) in vec3 initialPosition;
layout (location = 2) in vec3 positionOffset;

out vec4 fragColor;

void main()
{
  vec3 finalPosition = initialPosition + positionOffset;
  gl_Position = vec4(finalPosition, 1.0);
  fragColor = aColor;
}
