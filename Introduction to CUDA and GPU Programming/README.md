

# Introduction to CUDA and GPU Programming 

 <img src="https://github.com/user-attachments/assets/4ba2c60e-e699-4c4c-95e2-83499593f841" alt="CUDA Logo" width="300"/>

## In this tutorial, we’ll cover:

- What CUDA is and why it’s important
- Key concepts in CUDA
- Writing and running your first simple CUDA program

  Let’s dive in!

## Why Learn CUDA?
Parallel computing is essential in today’s world, where applications are growing more complex and data-heavy. Traditional CPUs process tasks sequentially, but GPUs can handle thousands of operations simultaneously. CUDA gives you access to this capability, allowing you to significantly accelerate the performance of applications like simulations, machine learning, and image processing.


## Real-World Applications 

- **Signal Processing and Communications:** Accelerating real-time processing for radar systems, communication protocols, and sensor networks. CUDA is often used to implement high-performance algorithms in areas like Fast Fourier Transforms (FFT) and filtering, essential for real-time data analysis.

- **Scientific Simulations:** Running complex models in physics, such as molecular dynamics or fluid simulations, where traditional CPUs can’t handle the volume of data or the speed required for accurate results.

- **Embedded Systems and Autonomous Systems:** In automotive or aerospace, CUDA allows for rapid prototyping of algorithms used in autonomous vehicles, robotics, and control systems. Tasks like image processing, sensor fusion, and path planning benefit from parallel execution on GPUs.

- **Advanced Data Analytics in Engineering:** Analyzing large-scale experimental data in research labs, where CUDA can significantly reduce the time needed for processing high-dimensional datasets. Whether it's statistical analysis or model fitting.

## Key Concepts in CUDA
Before jumping into coding, it’s crucial to understand the basic building blocks of CUDA programming:

1. Host and Device:

The Host refers to your CPU, while the Device refers to the GPU.
CUDA programming involves transferring data between the Host and Device.

2. Kernel Functions:

A kernel is a function that runs on the GPU. Kernels are executed by multiple threads in parallel.

3. Threads, Blocks, and Grids:

## CUDA uses a hierarchical structure:

- Threads: The smallest unit of execution.
- Blocks: Groups of threads.
- Grids: Groups of blocks.
  
You define how many threads, blocks, and grids you need based on the problem you’re solving.

![image](https://github.com/user-attachments/assets/043ed75a-2540-4d43-8874-a50ff8e80128)


## The CUDA Programming Model
The key advantage of CUDA is its ability to run thousands of threads concurrently. 
While this might sound complex, CUDA simplifies it by providing intuitive syntax and functions for managing threads, memory, and execution.

# Setting Up Your CUDA Environment

To setting up your  CUDA development environment please have look into the [ NVIDIA CUDA Installation Guide for Windows​](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html) & [NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)


# Writing Your First CUDA Program !

```cu
#include <iostream>
#include <cuda.h>

using namespace std;

// CUDA Kernel function to add elements of two arrays
__global__ void add(int *a, int *b, int *c) {
    int index = threadIdx.x;
    c[index] = a[index] + b[index];
}

int main() {
    // Array size
    int n = 10;
    int size = n * sizeof(int);

    // Host arrays
    int h_a[n], h_b[n], h_c[n];

    // Initialize arrays
    for(int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // Device arrays
    int *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Define the number of threads per block and the number of blocks using dim3 (x, y, z) 
    dim3 threadsPerBlock(n, 1, 1);
    dim3 blocksPerGrid(1, 1, 1);

    // Launch kernel on the GPU using dim3 configuration
    add<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c);

    // Copy result back to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Display the results
    for(int i = 0; i < n; i++) {
        cout << h_a[i] << " + " << h_b[i] << " = " << h_c[i] << endl;
    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}

```


