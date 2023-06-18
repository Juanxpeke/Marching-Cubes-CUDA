#version 460 core

uniform mat4 view;
uniform mat4 projection;

in vec3 position;

void main() {
	gl_Position = projection * view * vec4(position.x, position.y, position.z, 1.0);
}