#include "grid_renderer.h"

GridRenderer::GridRenderer(int cells, float size):
cells(cells),
size(size),
numVertices((cells + 1) * (cells + 1) * 6)
{
  // Near and far distance calculation
  near = cells > 8 ? (2 * size / cells) : 0.4f; 
  far = cells > 8 ? (4 * size / cells) : 3.2f;

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
  GLfloat* vertices = new GLfloat[numVertices * 4];

  float halfSize = size / 2.0f;
  float cellSize = size / cells;
  int idx = 0;

  // X axis parallel lines
  for (int i = 0; i < cells + 1; i++)
  {
    for(int j = 0; j < cells + 1; j++)
    {
      bool isBorder = (i == 0 || i == cells) && (j == 0 || j == cells);
      vertices[idx + 0] = -halfSize; // X left
      vertices[idx + 1] = -halfSize + i * cellSize; // Y
      vertices[idx + 2] = -halfSize + j * cellSize; // Z
      vertices[idx + 3] = isBorder ? 1.0f : 0.0f; // Border weight
      vertices[idx + 4] = halfSize; // X right
      vertices[idx + 5] = -halfSize + i * cellSize; // Y
      vertices[idx + 6] = -halfSize + j * cellSize; // Z
      vertices[idx + 7] = isBorder ? 1.0f : 0.0f; // Border weight
      idx += 8;
    }
  }
  // Y axis parallel lines
  for (int i = 0; i < cells + 1; i++)
  {
    for(int j = 0; j < cells + 1; j++)
    {
      bool isBorder = (i == 0 || i == cells) && (j == 0 || j == cells);
      vertices[idx + 0] = -halfSize + i * cellSize; // X
      vertices[idx + 1] = -halfSize; // Y bottom
      vertices[idx + 2] = -halfSize + j * cellSize; // Z
      vertices[idx + 3] = isBorder ? 1.0f : 0.0f; // Border weight
      vertices[idx + 4] = -halfSize + i * cellSize; // X
      vertices[idx + 5] = halfSize; // Y top
      vertices[idx + 6] = -halfSize + j * cellSize; // Z
      vertices[idx + 7] = isBorder ? 1.0f : 0.0f; // Border weight
      idx += 8;
    }
  }
  // Z axis parallel lines
  for (int i = 0; i < cells + 1; i++)
  {
    for(int j = 0; j < cells + 1; j++)
    {
      bool isBorder = (i == 0 || i == cells) && (j == 0 || j == cells);
      vertices[idx + 0] = -halfSize + j * cellSize; // X
      vertices[idx + 1] = -halfSize + i * cellSize; // Y
      vertices[idx + 2] = -halfSize; // Z back
      vertices[idx + 3] = isBorder ? 1.0f : 0.0f; // Border weight
      vertices[idx + 4] = -halfSize + j * cellSize; // X
      vertices[idx + 5] = -halfSize + i * cellSize; // Y
      vertices[idx + 6] = halfSize; // Z front
      vertices[idx + 7] = isBorder ? 1.0f : 0.0f; // Border weight
      idx += 8;
    }
  }

  // Generate the VAO and VBO
  glGenVertexArrays(1, &VAO);
  glGenBuffers(1, &VBO);

  glBindVertexArray(VAO);
  glBindBuffer(GL_ARRAY_BUFFER, VBO);

  glBufferData(GL_ARRAY_BUFFER, numVertices * 4 * sizeof(float), vertices, GL_STATIC_DRAW);
  glVertexAttribPointer(glGetAttribLocation(shaderProgram, "position"), 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*) 0);
  glEnableVertexAttribArray(glGetAttribLocation(shaderProgram, "position"));

  // Bind both the VBO and VAO to 0 so that we don't accidentally modify the VAO and VBO
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);

  delete vertices;
}

void GridRenderer::render()
{
  glUniform1f(glGetUniformLocation(shaderProgram, "near"), near);
  glUniform1f(glGetUniformLocation(shaderProgram, "far"), far);

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