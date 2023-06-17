#include <iostream>
#include <string>
#include <fstream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <glm/gtx/vector_angle.hpp>

// CUDA
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>

#include "defines.h"

// Util
#include "camera.h"
#include "performance_monitor.h"

// Kernels
extern "C" void launch_classifyVoxel(dim3 grid, dim3 threads, uint *voxelVerts,
                                     uint *voxelOccupied, uchar *volume,
                                     uint3 gridSize, uint3 gridSizeShift,
                                     uint3 gridSizeMask, uint numVoxels,
                                     float3 voxelSize, float isoValue);

extern "C" void launch_generateTriangles(
    dim3 grid, dim3 threads, float4 *pos, float4 *norm,
    uint *numVertsScanned, uint3 gridSize,
    uint3 gridSizeShift, uint3 gridSizeMask, float3 voxelSize, float isoValue,
    uint maxVerts, float xpos, float zpos);

extern "C" void allocateTextures(uint **d_edgeTable, uint **d_triTable,
                                 uint **d_numVertsTable);

extern "C" void destroyAllTextureObjects();

extern "C" void ThrustScanWrapper(unsigned int *output, unsigned int *input,
                                  unsigned int numElements);

// MC variables
uint3 gridSizeLog2 = make_uint3(5, 5, 5);
uint3 gridSizeShift;
uint3 gridSize;
uint3 gridSizeMask;

float3 voxelSize;
uint numVoxels = 0;
uint maxVerts = 0;
uint totalVerts = 0;

// Device data
GLuint posVbo, normalVbo;
GLint gl_Shader;
// CUDA-OpenGL interoperability
struct cudaGraphicsResource *cuda_posvbo_resource, *cuda_normalvbo_resource;  

float4 *d_pos = 0, *d_normal = 0;

uchar *d_volume = 0;
uint *d_voxelVerts = 0;
uint *d_voxelVertsScan = 0;
uint *d_voxelOccupied = 0;
uint *d_voxelOccupiedScan = 0;

// tables
uint *d_numVertsTable = 0;
uint *d_edgeTable = 0;
uint *d_triTable = 0;

float isoValue = 0.0f;
float dIsoValue = 0.01f;

// Rendering variables
glm::mat4 model = glm::mat4(1.0f);

Camera* camera;
PerformanceMonitor* performanceMonitor;

float lastTime;
float dt;

////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBO(GLuint *vbo, unsigned int size) {
  // create buffer object
  glGenBuffers(1, vbo);
  glBindBuffer(GL_ARRAY_BUFFER, *vbo);

  // initialize buffer object
  glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

// Initialize CUDA and MC stuff
void initCuda()
{
  gridSize = make_uint3(1 << gridSizeLog2.x, 1 << gridSizeLog2.y, 1 << gridSizeLog2.z);
  gridSizeMask = make_uint3(gridSize.x - 1, gridSize.y - 1, gridSize.z - 1);
  gridSizeShift = make_uint3(0, gridSizeLog2.x, gridSizeLog2.x + gridSizeLog2.y);

  numVoxels = gridSize.x * gridSize.y * gridSize.z;
  voxelSize = make_float3(2.0f / gridSize.x, 2.0f / gridSize.y, 2.0f / gridSize.z);
  maxVerts = gridSize.x * gridSize.y * 100;

  // Create VBOs
  createVBO(&posVbo, maxVerts * sizeof(float) * 4);
  cudaGraphicsGLRegisterBuffer(&cuda_posvbo_resource, posVbo, cudaGraphicsMapFlagsWriteDiscard);

  createVBO(&normalVbo, maxVerts * sizeof(float) * 4);
  cudaGraphicsGLRegisterBuffer(&cuda_normalvbo_resource, normalVbo, cudaGraphicsMapFlagsWriteDiscard);

  // Allocate textures
  allocateTextures(&d_edgeTable, &d_triTable, &d_numVertsTable);

  // Allocate device memory
  unsigned int memSize = sizeof(uint) * numVoxels;
  cudaMalloc((void **)&d_voxelVerts, memSize);
  cudaMalloc((void **)&d_voxelVertsScan, memSize);
  cudaMalloc((void **)&d_voxelOccupied, memSize);
  cudaMalloc((void **)&d_voxelOccupiedScan, memSize);
}

void computeIsosurface()
{
  int threads = 128;
  dim3 grid(numVoxels / threads, 1, 1);

  // get around maximum grid size of 65535 in each dimension
  if (grid.x > 65535) {
    grid.y = grid.x / 32768;
    grid.x = 32768;
  }

  // calculate number of vertices need per voxel
  launch_classifyVoxel(grid, threads, d_voxelVerts, d_voxelOccupied, d_volume,
                        gridSize, gridSizeShift, gridSizeMask, numVoxels,
                        voxelSize, isoValue);

  // scan voxel vertex count array
  ThrustScanWrapper(d_voxelVertsScan, d_voxelVerts, numVoxels);

  // readback total number of vertices
  {
    uint lastElement, lastScanElement;
    cudaMemcpy((void *)&lastElement,
                                (void *)(d_voxelVerts + numVoxels - 1),
                                sizeof(uint), cudaMemcpyDeviceToHost);
    cudaMemcpy((void *)&lastScanElement,
                                (void *)(d_voxelVertsScan + numVoxels - 1),
                                sizeof(uint), cudaMemcpyDeviceToHost);
    totalVerts = lastElement + lastScanElement;
  }

  // Generate triangles, writing to vertex buffers
  size_t num_bytes;
  cudaGraphicsMapResources(1, &cuda_posvbo_resource, 0);
  cudaGraphicsResourceGetMappedPointer((void **)&d_pos, &num_bytes, cuda_posvbo_resource);

  cudaGraphicsMapResources(1, &cuda_normalvbo_resource, 0);
  cudaGraphicsResourceGetMappedPointer((void **)&d_normal, &num_bytes, cuda_normalvbo_resource);

  dim3 grid2((int)ceil(numVoxels / (float)NTHREADS), 1, 1);

  while (grid2.x > 65535) {
    grid2.x /= 2;
    grid2.y *= 2;
  }

  launch_generateTriangles(grid2, NTHREADS, d_pos, d_normal,
                            d_voxelVertsScan, gridSize, gridSizeShift,
                            gridSizeMask, voxelSize, isoValue, maxVerts, 0.0f, 0.0f);

  cudaGraphicsUnmapResources(1, &cuda_normalvbo_resource, 0);
  cudaGraphicsUnmapResources(1, &cuda_posvbo_resource, 0);
}

// Function to handle mouse movement
void handleMouseMove(GLFWwindow* window, double xPos, double yPos) {
  camera->handleMouseMove(xPos, yPos);
}

// Function to handle mouse button events
void handleMouseClick(GLFWwindow* window, int button, int action, int mods) {
  camera->handleMouseClick(button, action, mods);
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

  int width = 1280;
  int height = 720;
  char title[256];

  // Creating a glfw window    
  GLFWwindow* window = glfwCreateWindow(width, height, title, NULL, NULL);

  if (window == NULL)
  {
    std::cout << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    exit(1);
  } else {
    std::cout << "GLFW window created" << std::endl;
  }

  glfwMakeContextCurrent(window);

  camera = new Camera(window, 1280, 720);
  performanceMonitor = new PerformanceMonitor(glfwGetTime(), 1.0f);

  // Set mouse movement callback
  glfwSetCursorPosCallback(window, handleMouseMove);

  // Set mouse button callback
  glfwSetMouseButtonCallback(window, handleMouseClick);
  
#if DISABLE_FPS_CAPPING
  // GLFW will swap buffers as soon as possible
  glfwSwapInterval(0);
#endif

  // Loading all OpenGL function pointers with glad
  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
  {
    std::cout << "Failed to initialize GLAD" << std::endl;
    exit(1);
  } else {
    std::cout << "GLAD initialized successfully" << std::endl;
  }

  // Shaders
  std::string vertexShaderCode = getFileContent("../../shaders/mc.vert");
	std::string fragmentShaderCode = getFileContent("../../shaders/mc.frag");

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
	GLuint VAO;

	// Generate the VAO and VBO with only 1 object each
	glGenVertexArrays(1, &VAO);

	// Make the VAO the current Vertex Array Object by binding it
	glBindVertexArray(VAO);

  initCuda();

  glBindBuffer(GL_ARRAY_BUFFER, posVbo);
	glVertexAttribPointer(glGetAttribLocation(shaderProgram, "position"), 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*) 0);
	glEnableVertexAttribArray(glGetAttribLocation(shaderProgram, "position"));

  glBindBuffer(GL_ARRAY_BUFFER, normalVbo);
  glVertexAttribPointer(glGetAttribLocation(shaderProgram, "normal"), 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*) 0);
	glEnableVertexAttribArray(glGetAttribLocation(shaderProgram, "normal"));

  glBindBuffer(GL_ARRAY_BUFFER, 0);

  glm::vec3 position = glm::vec3(0.0f, 0.0f, 0.0f);
	model = glm::translate(model, position);

	glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));

  glUniform3f(glGetUniformLocation(shaderProgram, "La"), 1.0f, 1.0f, 1.0f);
  glUniform3f(glGetUniformLocation(shaderProgram, "Ld"), 1.0f, 1.0f, 1.0f);

  glUniform3f(glGetUniformLocation(shaderProgram, "Ka"), 0.2f, 0.2f, 0.2f);
  glUniform3f(glGetUniformLocation(shaderProgram, "Kd"), 0.9f, 0.9f, 0.9f);

  glUniform3f(glGetUniformLocation(shaderProgram, "lightPosition"), 2.0f, 0.0f, 0.0f);
  
  glUniform1ui(glGetUniformLocation(shaderProgram, "shininess"), 100);
  glUniform1f(glGetUniformLocation(shaderProgram, "constantAttenuation"), 0.001f);
  glUniform1f(glGetUniformLocation(shaderProgram, "linearAttenuation"), 0.1f);
  glUniform1f(glGetUniformLocation(shaderProgram, "quadraticAttenuation"), 0.01f);

	// Specify the color of the background
	glClearColor(0.02f, 0.02f, 0.02f, 1.0f);
	// Clean the back buffer and assing the new color to it
	glClear(GL_COLOR_BUFFER_BIT);
  // Enables the Depth Buffer
	glEnable(GL_DEPTH_TEST);
	// Swap the back buffer with the front buffer
	glfwSwapBuffers(window);

  std::cout << "Opening GLFW window" << std::endl;

  while (!glfwWindowShouldClose(window))
  {
    // Using GLFW to check and process input events
    // internally, it stores all input events in the controller
    glfwPollEvents();

    performanceMonitor->update(glfwGetTime());

    sprintf(title, "CUDA Marching Cubes [%.1f FPS]", performanceMonitor->getFPS());
    glfwSetWindowTitle(window, title);

    camera->update(performanceMonitor->dt);
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(camera->view));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(camera->projection));
	
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Computes the isosurface
    computeIsosurface();

    glDrawArrays(GL_TRIANGLES, 0, totalVerts);

		glfwSwapBuffers(window);
  }
  
  std::cout << "GLFW window closed" << std::endl;

  return 0;
}