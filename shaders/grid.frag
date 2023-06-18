#version 460 core

uniform float renderDistance;
uniform vec3 viewPosition;

in vec3 fragPosition;

out vec4 fragColor;

void main()
{
  float dx = viewPosition.x - fragPosition.x;
  float dy = viewPosition.y - fragPosition.y;
  float dz = viewPosition.z - fragPosition.z;
  float distance = sqrt(dx * dx + dy * dy + dz * dz);

  float cappedDistance = min(distance, renderDistance);
  float alpha = 0.6f * (1.0f - cappedDistance / renderDistance); 

  fragColor = vec4(1.0, 1.0, 1.0, alpha);
}