#version 460 core

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

in vec4 position;
in vec4 normal;

out vec3 fragPosition;
out vec3 fragNormal;

void main() {
	gl_Position = projection * view * model * position;
  fragPosition = vec3(model * position);
  fragNormal = mat3(transpose(inverse(model))) * vec3(normal.x, normal.y, normal.z);  
}