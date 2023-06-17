#include "camera.h"

Camera::Camera(GLFWwindow* window, int width, int height, glm::vec3 position, glm::vec3 target):
window(window),
width(width),
height(height),
position(position),
targetOffset(target - position)
{
}

Camera::Camera(GLFWwindow* window, int width, int height):
window(window),
width(width),
height(height)
{  
}

void Camera::updateFromInputs()
{
  if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
    glm::vec3 front = glm::normalize(targetOffset);
    position += speed * front;
  }
  if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
    glm::vec3 right = glm::normalize(glm::cross(targetOffset, up));
    position -= speed * right;
  }
  if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
    glm::vec3 front = glm::normalize(targetOffset);
    position -= speed * front;
  }
  if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
    glm::vec3 right = glm::normalize(glm::cross(targetOffset, up));
    position += speed  * right;
	}
}

void Camera::update()
{
	updateFromInputs();

  view = glm::lookAt(position, position + targetOffset, up);
	projection = glm::perspective(glm::radians(FOV), (float)(width / height), near, far);
}

void Camera::handleMouseMove(double xPos, double yPos) {
  if (!isLeftButtonClicked) {
    mouseX = xPos;
    mouseY = yPos;
    return;
  }
  // Calculate the mouse movement deltas
  double deltaX = xPos - mouseX;
  double deltaY = yPos - mouseY;
  // Update the mouse position for the next frame
  mouseX = xPos;
  mouseY = yPos;
  // Update the camera angles based on the mouse movement
  targetTheta += sensitivity * deltaX;
  targetPhi -= sensitivity * deltaY;
  // Clamp the phi angle to avoid flipping the view
  targetPhi = glm::clamp(targetPhi, -1.5f, 1.5f);
  // Update the target offset vector
  targetOffset = glm::vec3(cos(targetTheta) * cos(targetPhi), sin(targetPhi), sin(targetTheta) * cos(targetPhi));
}

void Camera::handleMouseClick(int button, int action, int mods) {
  if (button == GLFW_MOUSE_BUTTON_LEFT) {
    if (action == GLFW_PRESS) {
      isLeftButtonClicked = true;
    } else if (action == GLFW_RELEASE) {
      isLeftButtonClicked = false;
    }
  }
}
