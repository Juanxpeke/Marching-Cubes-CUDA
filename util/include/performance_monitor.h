#pragma once

#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <string>
#include <chrono>
#include <map>
#include <vector>

class PerformanceMonitor
{
public:
	static const int MARCHING_CUBE_PROCESS = 0;
	static const int CLASSIFY_PROCESS = 1;
	static const int GENERATE_TRIANGLES_PROCESS = 2;
	static const int DRAW_PROCESS = 3;

	float dt;
	float framesPerSecond;

	PerformanceMonitor(float currentTime, const std::string &exportFolderName);
	void update(float currentTime); // Must be called each frame
	void startProcessTimer(int process);
  void endProcessTimer(int process);

private:
	float fpsTimer;
	int fpsResetIterations;
	int fpsPeriodIterations;

	int totalIterations;
	float lastIterationTime;

	std::chrono::high_resolution_clock::time_point processStartTimes[4];
  std::chrono::high_resolution_clock::time_point processEndTimes[4];
	
	int processIterations[4] = { 0, 0, 0, 0 };
	double processElapsedTimes[4] = { 0.0, 0.0, 0.0, 0.0 };
	std::string processKeys[4] = { "marching_cubes", "classify", "generate", "draw" };

	std::map<std::string,std::vector<std::pair<double, double>>> store;

	bool dataExported = false;
	std::string exportFolderName;
	int exportDataIterations;

	void exportData(const std::string &folder);
};