#pragma once

#include<glad/glad.h>
#include<GLFW/glfw3.h>
#include<glm/glm.hpp>
#include<glm/gtc/matrix_transform.hpp>
#include<glm/gtc/type_ptr.hpp>
#include<glm/gtx/rotate_vector.hpp>
#include<glm/gtx/vector_angle.hpp>

class Camera
{
public:

	// View variables
	glm::vec3 position;
	glm::vec3 target;
	glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);

	// Projection variables
	int width;
	int height;
	float FOV = 90.0f;
	float near = 0.1;
	float far = 100;

	// Projection x View matrix
	glm::mat4 cameraMatrix = glm::mat4(1.0f);

	// Movement
	float speed = 0.1f;
	float sensitivity = 100.0f;

  Camera(int width, int height, glm::vec3 position, glm::vec3 target);
	void processInputs(GLFWwindow* window);
	void update();
};