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
	static const int CLASSIFY_PROCESS = 0;
	static const int GENERATE_TRIANGLES_PROCESS = 1;
	static const int DRAW_PROCESS = 2;

	float dt;
	float framesPerSecond;

	PerformanceMonitor(float currentTime, const std::string &exportFolderName);
	void update(float currentTime);
	void startProcessTimer(int process);
  void endProcessTimer(int process);

private:
	float fpsPeriod;
	float fpsTimer;
	int fpsPeriodIterations;

	int totalIterations;
	float lastIterationTime;

	std::chrono::high_resolution_clock::time_point processStartTime;
  std::chrono::high_resolution_clock::time_point processEndTime;
	
	int processIterations[3] = { 0, 0, 0 };
	double processElapsedTimes[3] = { 0.0, 0.0, 0.0 };
	std::string processKeys[3] = { "classify", "generate", "draw" };

	std::map<std::string,std::vector<std::pair<double, double>>> store;

	bool dataExported = false;
	std::string exportFolderName;
	int exportDataIterations = 1000;

	void exportData(const std::string &folder);
};