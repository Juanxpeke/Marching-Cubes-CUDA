#include "camera.h"

Camera::Camera(int width, int height, glm::vec3 position, glm::vec3 target):
width(width),
height(height),
position(position),
target(target)
{
}

void Camera::processInputs(GLFWwindow* window)
{
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
		position += speed * glm::normalize(target - position);
	}
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
		position -= speed * glm::normalize(glm::cross(target - position, up));
	}
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
		position -= speed * glm::normalize(target - position);
	}
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
		position += speed * glm::normalize(glm::cross(target - position, up));
	}
}

void Camera::update()
{
  glm::mat4 view = glm::lookAt(position, target, up);
	glm::mat4 projection = glm::perspective(glm::radians(FOV), (float)(width / height), near, far);

	cameraMatrix = projection * view;
}
