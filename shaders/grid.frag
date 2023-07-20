#version 460 core

uniform float near;
uniform float far;
uniform vec3 viewPosition;

in vec3 fragPosition;
in float fragBorderWeight;

out vec4 fragColor;

void main()
{
  float fDistance = distance(viewPosition, fragPosition);

  float cappedDistance = max(min(fDistance, far), near);

  float halfDistance = (near + far) * 0.5f;
  float halfLength = (far - near) * 0.5f; 
  float alpha = 0.4f * (1.0f - abs(cappedDistance - halfDistance) / halfLength) + fragBorderWeight; 

  fragColor = vec4(1.0, 1.0, 1.0, alpha);
}