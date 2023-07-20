#version 460 core

uniform mat4 view;
uniform mat4 projection;

in vec4 position;

out vec3 fragPosition;
out float fragBorderWeight;

void main() {
	gl_Position = projection * view * vec4(position.x, position.y, position.z, 1.0);
	fragPosition = vec3(position.x, position.y, position.z);
	fragBorderWeight = position.w;
}