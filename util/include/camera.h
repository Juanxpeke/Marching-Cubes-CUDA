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
	glm::vec3 position = glm::vec3(0.0f, 2.0f, -2.0f);
	glm::vec3 targetOffset = glm::vec3(0.0f, -2.0f, 2.0f);;
	glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
	glm::mat4 view = glm::mat4(1.0f);

	// Projection variables
	int width;
	int height;
	float FOV = 60.0f;
	float near = 0.0001f;
	float far = 100.0f;
	glm::mat4 projection = glm::mat4(1.0f);

	// Movement
	float speed = 2.0f;
	float sensitivity = 0.01f;

  Camera(GLFWwindow* window, int width, int height, glm::vec3 position, glm::vec3 target);
	Camera(GLFWwindow* window, int width, int heigth);
	void update(float dt);

	// Input callbacks
	void handleMouseMove(double xPos, double yPos);
	void handleMouseClick(int button, int action, int mods);

private:
	// Window
	GLFWwindow* window;

	// Input variables
	double mouseX = 0.0;
	double mouseY = 0.0;

	float targetTheta = 3.1415f / 2;
	float targetPhi = -3.1415 / 4;

	bool isLeftButtonClicked = false;
	
	// Update variables from keyboard inputs
	void updateFromInputs(float dt);
};