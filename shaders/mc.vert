#version 460 core

uniform mat4 view;
uniform mat4 projection;

in vec4 position;
in vec4 normal;

out vec3 fragPosition;
out vec3 fragNormal;

void main() {
	gl_Position = projection * view * position;
  fragPosition = vec3(position);
  fragNormal = vec3(normal);  
}