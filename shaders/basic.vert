#version 460 core

uniform mat4 model;
uniform mat4 camera;

in vec3 position;

void main() {
	gl_Position = camera * model * vec4(position.x, position.y, position.z, 1.0);
}