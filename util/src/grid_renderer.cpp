#include "grid_renderer.h"

GridRenderer::GridRenderer(int cells, float gridSize):
cells(cells),
gridSize(gridSize),
numVertices((cells + 1) * (cells + 1) * 6)
{
  // Enable blending
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  compileShaders();
  initBuffers();
}

void GridRenderer::compileShaders()
{
  std::string vertexShaderCode = getFileContent("../../shaders/grid.vert");
  std::string fragmentShaderCode = getFileContent("../../shaders/grid.frag");

  const char* vertexShaderSource = vertexShaderCode.c_str();
  const char* fragmentShaderSource = fragmentShaderCode.c_str();

  // Create Vertex Shader Object and get its reference
  GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
  glCompileShader(vertexShader);

  // Create Fragment Shader Object and get its reference
  GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
  glCompileShader(fragmentShader);

  // Create Shader Program Object and get its reference
  shaderProgram = glCreateProgram();
  glAttachShader(shaderProgram, vertexShader);
  glAttachShader(shaderProgram, fragmentShader);

  // Wrap-up / link all the shaders together into the Shader Program
  glLinkProgram(shaderProgram);

  // Delete the now useless Vertex and Fragment Shader Objects
  glDeleteShader(vertexShader);
  glDeleteShader(fragmentShader);
}

void GridRenderer::initBuffers()
{
  GLfloat* vertices = new GLfloat[numVertices * 3];

  float gridHalf = gridSize / 2.0f;
  float voxelSize = gridSize / cells;
  int idx = 0;

  // X axis parallel lines
  for (int i = 0; i < cells + 1; i++)
  {
    for(int j = 0; j < cells + 1; j++)
    {
      vertices[idx + 0] = -gridHalf; // X left
      vertices[idx + 1] = -gridHalf + i * voxelSize; // Y
      vertices[idx + 2] = -gridHalf + j * voxelSize; // Z
      vertices[idx + 3] = gridHalf; // X right
      vertices[idx + 4] = -gridHalf + i * voxelSize; // Y
      vertices[idx + 5] = -gridHalf + j * voxelSize; // Z
      idx += 6;
    }
  }
  // Y axis parallel lines
  for (int i = 0; i < cells + 1; i++)
  {
    for(int j = 0; j < cells + 1; j++)
    {
      vertices[idx + 0] = -gridHalf + i * voxelSize; // X
      vertices[idx + 1] = -gridHalf; // Y bottom
      vertices[idx + 2] = -gridHalf + j * voxelSize; // Z
      vertices[idx + 3] = -gridHalf + i * voxelSize; // X
      vertices[idx + 4] = gridHalf; // Y top
      vertices[idx + 5] = -gridHalf + j * voxelSize; // Z
      idx += 6;
    }
  }
  // Z axis parallel lines
  for (int i = 0; i < cells + 1; i++)
  {
    for(int j = 0; j < cells + 1; j++)
    {
      vertices[idx + 0] = -gridHalf + j * voxelSize; // X
      vertices[idx + 1] = -gridHalf + i * voxelSize; // Y
      vertices[idx + 2] = -gridHalf; // Z back
      vertices[idx + 3] = -gridHalf + j * voxelSize; // X
      vertices[idx + 4] = -gridHalf + i * voxelSize; // Y
      vertices[idx + 5] = gridHalf; // Z front
      idx += 6;
    }
  }

  // Generate the VAO and VBO
  glGenVertexArrays(1, &VAO);
  glGenBuffers(1, &VBO);

  glBindVertexArray(VAO);
  glBindBuffer(GL_ARRAY_BUFFER, VBO);

  glBufferData(GL_ARRAY_BUFFER, numVertices * 3 * sizeof(float), vertices, GL_STATIC_DRAW);
  glVertexAttribPointer(glGetAttribLocation(shaderProgram, "position"), 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*) 0);
  glEnableVertexAttribArray(glGetAttribLocation(shaderProgram, "position"));

  // Bind both the VBO and VAO to 0 so that we don't accidentally modify the VAO and VBO
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);

  delete vertices;
}

void GridRenderer::render()
{
  glBindVertexArray(VAO);
  glDrawArrays(GL_LINES, 0, numVertices);
  glBindVertexArray(0);
}

GridRenderer::~GridRenderer()
{
  // Delete all the objects we've created
  glDeleteVertexArrays(1, &VAO);
  glDeleteBuffers(1, &VBO);
  glDeleteProgram(shaderProgram);
}