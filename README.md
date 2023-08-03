# Marching-Cubes-CUDA
 Marching Cubes implementation using CUDA and OpenGL. It is based on the CUDA samples marching cubes implementation, and extended for terrain generation.


## Observations 

With a NVIDIA GTX1660 GPU, a Intel Core i5-8600K CPU and 16GB of RAM.

Using ```gridLogSize(8, 8, 8)``` and no shared memory the average FPS is 23, taking around 45 seconds to run 1000 iterations. 

- ```ThrustScanWrapper()``` takes all this time if we don't use ```cudaDeviceSynchronize()```

- If we omit the previous call, the FPS are almost the same, and the total time for 1000 iterations is around 42 seconds

- As a conclusion, it seems the kernels logic was being executed when calling ```ThrustScanWrapper()```

- If we use ```cudaDeviceSynchronize()``` after classify kernel call, the total time now is used by this call

- If we use ```cudaDeviceSynchronize()``` after both classify and generate triangles kernels calls, the total time of classify is around 5 seconds and for generate triangles about 37 seconds

- Using shared memory, the classify kernel performance is the same, but generate triangles performance improves

Using ```gridLogSize(7, 7, 7)``` and no shared memory the average FPS is 130, taking around 8 seconds to run 1000 iterations.

- Without FPS capping, classify and generate triangles total time is almost the same as the entire CUDA compute surfaces call (around 8 seconds)

- With FPS capping (60 FPS), classify and generate triangles total time is the same as before, but the time it takes for the entire CUDA compute surafces call is now around 15 seconds

