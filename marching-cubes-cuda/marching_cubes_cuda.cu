#include <iostream>
#include <string>
#include <fstream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

// CUDA
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>

// Mesh width and height
const unsigned int meshWidth = 1024;
 
// Cuda Graphics Resource pointer
struct cudaGraphicsResource *cudaVBOResource;

// Animation time
float animTime = 0.0;

// ===================================
// ======== Density functions ========
// ===================================

float densityBase(float3 ws)
{
  float density = -ws.y;

  return density; 
}

// ========================================================
// ======== Cuda Kernel to modify vertex positions ========
// ========================================================
__global__ void vbo_kernel(float3 *pos, unsigned int width, float time)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate UV coordinates
    float u = x / (float) width;
    u = u * 2.0f - 1.0f;

    // Calculate simple wave pattern
    float freq = 8.0f;
    float v = cosf(u * freq + time);

    u = u * 0.5f; v = v * 0.5f;

    // write output vertex
    pos[x] = make_float3(u, v, 0.0f);
}

void runCuda(struct cudaGraphicsResource **cudaVBOResourcePointer)
{
  // Map OpenGL buffer object for writing from CUDA
  float3 *dptr;
  cudaGraphicsMapResources(1, cudaVBOResourcePointer, 0);
  size_t numBytes;
  cudaGraphicsResourceGetMappedPointer((void**) &dptr, &numBytes, *cudaVBOResourcePointer);
  
  // printf("CUDA mapped VBO: May access %ld bytes\n", num_bytes);

  // Block size
  int blockSize = 64;

  // Round up in case N is not a multiple of blockSize
  int numBlocks = (meshWidth + blockSize - 1) / blockSize;

  // Execute the kernel
  vbo_kernel<<<numBlocks, blockSize>>>(dptr, meshWidth, animTime);

  // Unmap buffer object
  cudaGraphicsUnmapResources(1, cudaVBOResourcePointer, 0);
}

std::string getFileContent(const char* filename)
{
  std::ifstream file(filename);

  if (!file) {
    std::cout << "Error reading the file " << filename;
    exit(0);
  }

  if (file.is_open())
  {
    std::string content((std::istreambuf_iterator<char>(file)),
                        (std::istreambuf_iterator<char>()));

    file.close();

    return content;
  } else {
    std::cout << "Failed to open the file " << filename << std::endl;
    exit(1);
  }
}

int main()
{
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

  // Creating a glfw window    
  GLFWwindow* glfwWindow = glfwCreateWindow(1280, 720, "Test OpenGL", NULL, NULL);

  if (glfwWindow == NULL)
  {
    std::cout << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    exit(1);
  } else {
    std::cout << "GLFW window created" << std::endl;
  }

  glfwMakeContextCurrent(glfwWindow);

  // Loading all OpenGL function pointers with glad
  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
  {
    std::cout << "Failed to initialize GLAD" << std::endl;
    exit(1);
  } else {
    std::cout << "GLAD initialized successfully" << std::endl;
  }

  // Shaders
  std::string vertexShaderCode = getFileContent("G:/Assets/Shaders/basic.vert");
	std::string fragmentShaderCode = getFileContent("G:/Assets/Shaders/basic.frag");

  const char* vertexShaderSource = vertexShaderCode.c_str();
	const char* fragmentShaderSource = fragmentShaderCode.c_str();

	// Create Vertex Shader Object and get its reference
	GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
	// Attach Vertex Shader source to the Vertex Shader Object
	glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
	// Compile the Vertex Shader into machine code
	glCompileShader(vertexShader);

	// Create Fragment Shader Object and get its reference
	GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	// Attach Fragment Shader source to the Vertex Shader Object
	glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
	// Compile the Fragment Shader into machine code
	glCompileShader(fragmentShader);

	// Create Shader Program Object and get its reference
	GLuint shaderProgram = glCreateProgram();
	// Attach the Vertex and Fragment Shaders to the Shader Program
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	// Wrap-up / link all the shaders together into the Shader Program
	glLinkProgram(shaderProgram);

	// Delete the now useless Vertex and Fragment Shader Objects
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

  // Use the shader program
  glUseProgram(shaderProgram);

  // Create reference containers for the Vertex Array Object and the Vertex Buffer Object
	GLuint VAO, VBO;

	// Generate the VAO and VBO with only 1 object each
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);

	// Make the VAO the current Vertex Array Object by binding it
	glBindVertexArray(VAO);

	// Bind the VBO specifying it's a GL_ARRAY_BUFFER
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	// Introduce the vertices into the VBO
	glBufferData(GL_ARRAY_BUFFER, meshWidth * 3 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

	// Configure the Vertex Attribute so that OpenGL knows how to read the VBO
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	// Enable the Vertex Attribute so that OpenGL knows to use it
	glEnableVertexAttribArray(0);

	// Bind both the VBO and VAO to 0 so that we don't accidentally modify the VAO and VBO
	glBindBuffer(GL_ARRAY_BUFFER, 0);

  // Register this buffer object with CUDA
  cudaGraphicsGLRegisterBuffer(&cudaVBOResource, VBO, cudaGraphicsMapFlagsWriteDiscard);

	// Specify the color of the background
	glClearColor(0.02f, 0.02f, 0.02f, 1.0f);
	// Clean the back buffer and assing the new color to it
	glClear(GL_COLOR_BUFFER_BIT);
	// Swap the back buffer with the front buffer
	glfwSwapBuffers(glfwWindow);

  std::cout << "Opening GLFW window" << std::endl;

  while (!glfwWindowShouldClose(glfwWindow))
  {
    // Run CUDA kernel to generate vertex positions
    runCuda(&cudaVBOResource);

    // Using GLFW to check and process input events
    // internally, it stores all input events in the controller
    glfwPollEvents();
	
		glClear(GL_COLOR_BUFFER_BIT);

		// Draw the triangle using the GL_POINTS primitive
		glDrawArrays(GL_POINTS, 0, meshWidth);
		glfwSwapBuffers(glfwWindow);

    // Update animation
    animTime += 0.04f;
  }
  
  std::cout << "GLFW window closed" << std::endl;

  return 0;
}