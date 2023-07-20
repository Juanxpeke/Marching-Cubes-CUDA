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

// Util
#include "camera.h"
#include "performance_monitor.h"
#include "file_manager.h"
#include "grid_renderer.h"

// Configuration macros
#include "defines.h"

// Kernels
extern "C" void launchClassifyVoxel(
  dim3 blocks, dim3 threads,
  uint *voxelVerts, uint *voxelOccupied,
  uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask,
  uint numVoxels, float3 voxelSize, float isoValue);

extern "C" void launchGenerateTriangles(
  dim3 blocks, dim3 threads,
  float4 *pos, float4 *norm,
  uint *numVertsScanned,
  uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask,
  float3 voxelSize, float isoValue, uint maxVerts);

extern "C" void allocateTextures(uint **dEdgeTable, uint **dTriTable,
                                 uint **dNumVertsTable);

extern "C" void destroyAllTextureObjects();

extern "C" void ThrustScanWrapper(unsigned int *output, unsigned int *input,
                                  unsigned int numElements);

// MC variables
uint3 gridSizeLog2 = make_uint3(5, 5, 5);
uint3 gridSizeShift;
uint3 gridSize;
uint3 gridSizeMask;

float worldSize = 20.0f;

float3 voxelSize;
uint numVoxels = 0;
uint maxVerts = 0;
uint totalVerts = 0;

float isoValue = 0.0f;

// OpenGL
GLuint posVbo, normalVbo;

// CUDA-OpenGL interoperability
struct cudaGraphicsResource *cudaPosVBOResource, *cudaNormalVBOResource;  
float4 *dPos = 0, *dNormal = 0;

// Device data
uint *dVoxelVerts = 0;
uint *dVoxelVertsScan = 0;
uint *dVoxelOccupied = 0;
uint *dVoxelOccupiedScan = 0;

// Tables
uint *dNumVertsTable = 0;
uint *dEdgeTable = 0;
uint *dTriTable = 0;

// Rendering variables
Camera* camera;
PerformanceMonitor* performanceMonitor;

// Creates the VBO
void createVBO(GLuint *vbo, unsigned int size) {
  // Create buffer object
  glGenBuffers(1, vbo);
  glBindBuffer(GL_ARRAY_BUFFER, *vbo);

  // Initialize buffer object
  glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

// Initialize marching cubes stuff
void initMarchingCubes()
{
  gridSize = make_uint3(1 << gridSizeLog2.x, 1 << gridSizeLog2.y, 1 << gridSizeLog2.z);
  gridSizeMask = make_uint3(gridSize.x - 1, gridSize.y - 1, gridSize.z - 1);
  gridSizeShift = make_uint3(0, gridSizeLog2.x, gridSizeLog2.x + gridSizeLog2.y);

  numVoxels = gridSize.x * gridSize.y * gridSize.z;
  voxelSize = make_float3(worldSize / gridSize.x, worldSize / gridSize.y, worldSize / gridSize.z);
  maxVerts = gridSize.x * gridSize.y * 100;

  // Create VBOs
  createVBO(&posVbo, maxVerts * sizeof(float) * 4);
  cudaGraphicsGLRegisterBuffer(&cudaPosVBOResource, posVbo, cudaGraphicsMapFlagsWriteDiscard);

  createVBO(&normalVbo, maxVerts * sizeof(float) * 4);
  cudaGraphicsGLRegisterBuffer(&cudaNormalVBOResource, normalVbo, cudaGraphicsMapFlagsWriteDiscard);

  // Allocate textures
  allocateTextures(&dEdgeTable, &dTriTable, &dNumVertsTable);

  // Allocate device memory
  unsigned int memSize = sizeof(uint) * numVoxels;
  cudaMalloc((void**) &dVoxelVerts, memSize);
  cudaMalloc((void**) &dVoxelVertsScan, memSize);
  cudaMalloc((void**) &dVoxelOccupied, memSize);
  cudaMalloc((void**) &dVoxelOccupiedScan, memSize);
}

void computeIsosurface()
{

  int threads = min(NTHREADS, numVoxels);
  dim3 blocks(numVoxels / threads, 1, 1);

  // Get around maximum grid size of 65535 in each dimension
  if (blocks.x > 65535) {
    blocks.y = blocks.x / 32768;
    blocks.x = 32768;
  }

  performanceMonitor->startProcessTimer(PerformanceMonitor::CLASSIFY_PROCESS);
  // Calculate number of vertices need per voxel
  launchClassifyVoxel(
    blocks, threads,
    dVoxelVerts, dVoxelOccupied,
    gridSize, gridSizeShift, gridSizeMask,
    numVoxels, voxelSize, isoValue);
  cudaDeviceSynchronize();
  performanceMonitor->endProcessTimer(PerformanceMonitor::CLASSIFY_PROCESS);  

  // Scan voxel vertex count array
  ThrustScanWrapper(dVoxelVertsScan, dVoxelVerts, numVoxels);

  // Readback total number of vertices
  {
    uint lastElement, lastScanElement;
    cudaMemcpy((void*) &lastElement, (void*) (dVoxelVerts + numVoxels - 1), sizeof(uint), cudaMemcpyDeviceToHost);
    cudaMemcpy((void*) &lastScanElement, (void*) (dVoxelVertsScan + numVoxels - 1), sizeof(uint), cudaMemcpyDeviceToHost);
    totalVerts = lastElement + lastScanElement;
  }

  // Generate triangles, writing to vertex buffers
  size_t numBytes;
  cudaGraphicsMapResources(1, &cudaPosVBOResource, 0);
  cudaGraphicsResourceGetMappedPointer((void **) &dPos, &numBytes, cudaPosVBOResource);

  cudaGraphicsMapResources(1, &cudaNormalVBOResource, 0);
  cudaGraphicsResourceGetMappedPointer((void **) &dNormal, &numBytes, cudaNormalVBOResource);

  dim3 grid2((int) ceil(numVoxels / (float) NTHREADS), 1, 1);

  while (grid2.x > 65535) {
    grid2.x /= 2;
    grid2.y *= 2;
  }

  performanceMonitor->startProcessTimer(PerformanceMonitor::GENERATE_TRIANGLES_PROCESS);
  launchGenerateTriangles(
    grid2, NTHREADS,
    dPos, dNormal,
    dVoxelVertsScan,
    gridSize, gridSizeShift, gridSizeMask,
    voxelSize, isoValue, maxVerts);
  cudaDeviceSynchronize();
  performanceMonitor->endProcessTimer(PerformanceMonitor::GENERATE_TRIANGLES_PROCESS);

  cudaGraphicsUnmapResources(1, &cudaNormalVBOResource, 0);
  cudaGraphicsUnmapResources(1, &cudaPosVBOResource, 0);
}

// Function to handle mouse movement
void handleMouseMove(GLFWwindow* window, double xPos, double yPos) {
  camera->handleMouseMove(xPos, yPos);
}

// Function to handle mouse button events
void handleMouseClick(GLFWwindow* window, int button, int action, int mods) {
  camera->handleMouseClick(button, action, mods);
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
  }

  glfwMakeContextCurrent(window);

  camera = new Camera(window, width, height);
  performanceMonitor = new PerformanceMonitor(glfwGetTime(), "mc-cuda");

  // Set GLFW callbacks
  glfwSetCursorPosCallback(window, handleMouseMove);
  glfwSetMouseButtonCallback(window, handleMouseClick);
  
#if DISABLE_FPS_CAPPING
  // GLFW will swap buffers as soon as possible
  glfwSwapInterval(0);
#endif

  // Loading all OpenGL function pointers with glad
  if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress))
  {
    std::cout << "Failed to initialize GLAD" << std::endl;
    exit(1);
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

  initMarchingCubes();

  glBindBuffer(GL_ARRAY_BUFFER, posVbo);
	glVertexAttribPointer(glGetAttribLocation(shaderProgram, "position"), 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*) 0);
	glEnableVertexAttribArray(glGetAttribLocation(shaderProgram, "position"));

  glBindBuffer(GL_ARRAY_BUFFER, normalVbo);
  glVertexAttribPointer(glGetAttribLocation(shaderProgram, "normal"), 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*) 0);
	glEnableVertexAttribArray(glGetAttribLocation(shaderProgram, "normal"));

  glBindBuffer(GL_ARRAY_BUFFER, 0);

  glUniform3f(glGetUniformLocation(shaderProgram, "La"), 1.0f, 1.0f, 1.0f);
  glUniform3f(glGetUniformLocation(shaderProgram, "Ld"), 1.0f, 1.0f, 1.0f);

  glUniform3f(glGetUniformLocation(shaderProgram, "Ka"), 0.2f, 0.2f, 0.2f);
  glUniform3f(glGetUniformLocation(shaderProgram, "Kd"), 0.9f, 0.9f, 0.9f);

  glUniform3f(glGetUniformLocation(shaderProgram, "lightPosition"), 4.0f, 4.0f, 4.0f);
  
  glUniform1ui(glGetUniformLocation(shaderProgram, "shininess"), 100);
  glUniform1f(glGetUniformLocation(shaderProgram, "constantAttenuation"), 0.001f);
  glUniform1f(glGetUniformLocation(shaderProgram, "linearAttenuation"), 0.01f);
  glUniform1f(glGetUniformLocation(shaderProgram, "quadraticAttenuation"), 0.0001f);

  glBindVertexArray(0);

  GridRenderer gridRenderer(gridSize.x, worldSize);

	// Specify the color of the background
	glClearColor(0.02f, 0.02f, 0.02f, 1.0f);
	// Clean the back buffer and assing the new color to it
	glClear(GL_COLOR_BUFFER_BIT);
  // Enables the depth Buffer
	glEnable(GL_DEPTH_TEST);
	// Swap the back buffer with the front buffer
	glfwSwapBuffers(window);

  while (!glfwWindowShouldClose(window))
  {
    // Using GLFW to check and process input events
    glfwPollEvents();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    performanceMonitor->update(glfwGetTime());
    camera->update(performanceMonitor->dt);

    sprintf(title, "CUDA Marching Cubes [%.1f FPS]", performanceMonitor->framesPerSecond);
    glfwSetWindowTitle(window, title);

    glUseProgram(shaderProgram);
    glBindVertexArray(VAO);

    performanceMonitor->startProcessTimer(PerformanceMonitor::MARCHING_CUBE_PROCESS);
    computeIsosurface();
    performanceMonitor->endProcessTimer(PerformanceMonitor::MARCHING_CUBE_PROCESS);

    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(camera->view));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(camera->projection));

    performanceMonitor->startProcessTimer(PerformanceMonitor::DRAW_PROCESS);
    glDrawArrays(GL_TRIANGLES, 0, totalVerts);
    performanceMonitor->endProcessTimer(PerformanceMonitor::DRAW_PROCESS);

    glUseProgram(gridRenderer.shaderProgram);
    glUniform3fv(glGetUniformLocation(gridRenderer.shaderProgram, "viewPosition"), 1, glm::value_ptr(camera->position));
    glUniformMatrix4fv(glGetUniformLocation(gridRenderer.shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(camera->view));
    glUniformMatrix4fv(glGetUniformLocation(gridRenderer.shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(camera->projection));
    gridRenderer.render();

		glfwSwapBuffers(window);
  }

  return 0;
}