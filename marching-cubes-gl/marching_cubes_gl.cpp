#include <iostream>
#include "renderer.h"

int main()
{
	Renderer renderer;

	renderer.init();

	renderer.compileShaders("../../shaders/basic.vert", "../../shaders/basic.frag");
	renderer.useShaders();
	
	// Specify the color of the background
	glClearColor(0.02f, 0.02f, 0.02f, 1.0f);
	// Clean the back buffer and assing the new color to it
	glClear(GL_COLOR_BUFFER_BIT);
	// Swap the back buffer with the front buffer
	glfwSwapBuffers(renderer.window);

  std::cout << "Opening GLFW window" << std::endl;

  while (!glfwWindowShouldClose(renderer.window))
  {
    // Using GLFW to check and process input events
    // internally, it stores all input events in the controller
    glfwPollEvents();
	
		renderer.render();
		
		glfwSwapBuffers(renderer.window);
  }
  
  std::cout << "GLFW window closed" << std::endl;

  return 0;
}