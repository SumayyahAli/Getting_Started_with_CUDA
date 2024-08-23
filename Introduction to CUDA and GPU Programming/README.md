# 1. Introduction to CUDA C/C++ and GPU for Prallal Computing 

## Overview

This guide will introduce you to:

- The basics of CUDA and why it’s significant
- Core concepts in CUDA programming
- How to write and run your first simple CUDA program

Let’s dive in!

## Why Learn CUDA?

In today's world, applications are increasingly data-heavy and complex. Traditional CPUs process tasks sequentially, which limits performance for parallel workloads. GPUs, with their thousands of cores, can execute many tasks simultaneously, making them ideal for high-performance applications like simulations, machine learning, and image processing. CUDA unlocks the full potential of GPU parallelism, enabling substantial acceleration in computing.

## Real-World Applications

- **Signal Processing and Communications:** CUDA boosts performance in real-time tasks like radar processing, communication protocols, and sensor networks by enabling fast and efficient algorithms such as FFTs and filtering.

- **Embedded and Autonomous Systems:** CUDA powers applications in automotive and aerospace by enabling rapid prototyping and real-time processing for tasks like sensor fusion and control systems.

- **Video and Image Processing:** From smoother streaming and faster video editing to enhanced image analysis, CUDA accelerates tasks like real-time encoding, decoding, and object detection, making it essential for AR, computer vision, and surveillance systems.

- **Scientific Simulations:** Whether it’s simulating molecular interactions or large-scale physical models, CUDA handles the heavy lifting where traditional CPUs fall short in speed and scale.

- **Advanced Data Analytics:** For research labs dealing with massive datasets, CUDA speeds up statistical modeling, experimental data analysis, and other high-dimensional data tasks, delivering results faster and more efficiently.

## Basic Concepts in CUDA

Before you start coding, it’s important to grasp some basic CUDA concepts:

1. **Host and Device:**  
   The Host refers to the CPU, while the Device refers to the GPU. CUDA programming involves data transfer between the Host and Device.

2. **Kernel Functions:**  
   A kernel is a function that runs on the GPU, executed in parallel by multiple threads.

3. **Threads, Blocks, and Grids:**  
   CUDA follows a hierarchical structure:
   - **Threads:** The smallest unit of execution.
   - **Blocks:** Groups of threads.
   - **Grids:** Groups of blocks.
   
   You’ll need to define the number of threads, blocks, and grids based on the problem you’re solving.

<img width="800" alt="image" src="https://github.com/user-attachments/assets/043ed75a-2540-4d43-8874-a50ff8e80128">

## The CUDA Programming Model

CUDA excels at managing thousands of concurrent threads while keeping code straightforward. It provides intuitive syntax for handling threads, memory, and execution allowing us to focus on writing efficient algorithms.

## Setting Up Your CUDA Environment

To get started with CUDA development, follow the installation guides for your operating system:
- [NVIDIA CUDA Installation Guide for Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)
- [NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

## Writing Your First CUDA Program

Clone the source code for your first CUDA program:

**(I'm using CUDA with Visual Studio on Windows OS)**

```
git clone https://github.com/your-repo/first-cuda-program.git
```

# Writing Your First CUDA Program !

clone source code `/My_First_CUDA.cu`

`My_First_CUDA.cu`:
```cu
#include <iostream>
// in Windows 
#include <cuda_runtime.h>

using namespace std;

// CUDA Kernel function to add elements of two arrays
__global__ void add(int* a, int* b, int* c) {
    int index = threadIdx.x;
    c[index] = a[index] + b[index];
}

int main() {
    // Array size
    const int n = 10;
    int size = n * sizeof(int);

    // Host arrays
    int h_a[n], h_b[n], h_c[n];

    // Initialize arrays
    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // Device arrays
    int* d_a, * d_b, * d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    /* Define the number of threads per block and the number of blocks using dim3
       *** What is dim3? **
         The dim3 data type in CUDA is used to define the dimensions of blocks and grids.
         It allows you to specify the number of threads in each block and the number of blocks in each grid.
          You can think of dim3 as a 3D vector with x, y, and z dimensions. In most simple cases  */

    dim3 threadsPerBlock(n, 1, 1);
    dim3 blocksPerGrid(1, 1, 1);

    // Launch kernel on the GPU using dim3 configuration
    add << <blocksPerGrid, threadsPerBlock >> > (d_a, d_b, d_c);

    // Copy result back to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Display the results
    for (int i = 0; i < n; i++) {
        cout << h_a[i] << " + " << h_b[i] << " = " << h_c[i] << endl;
    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
```
## How our Code Works?:

![image](https://github.com/user-attachments/assets/bc447e36-7807-4e9d-8d2f-6232f9addf8f)

**1. Kernel Function (`add`):**

The `__global__` keyword defines the add function as a CUDA kernel, meaning it runs directly on the GPU. Each thread operates on one element of the arrays, adding corresponding values from a and b and storing the result in c.

**2. Memory Management:** 

  Memory is allocated on both the host (CPU) and device (GPU) using cudaMalloc. The data is then copied from the host arrays (`h_a`, `h_b`) to the device arrays (`d_a`, `d_b`) using `cudaMemcpy`.

**3. Grid and Block Configuration:**

The kernel is launched with a configuration defined by `dim3`:
- `dim3 threadsPerBlock(n, 1, 1)`; sets n threads in a block (each thread handles one element).
- `dim3 blocksPerGrid(1, 1, 1)`; specifies a single block in the grid (since our data is small).
This setup ensures that all n elements are processed in parallel.

**4. Kernel Launch:**

 The kernel is launched using `add<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c);`. This syntax specifies how the work is distributed across the GPU.

**5. Copied Results and Cleanup:**

After computation, the results are copied back to the host, displayed, and the device memory is freed using `cudaFree`.

## Output
This shows that the arrays are added correctly in parallel.

<img width="230" alt="image" src="https://github.com/user-attachments/assets/98f6adfa-75ab-46a6-b3ce-3806153f1f17">

## Summary

- **Why CUDA is Important:** We explored how GPUs are built for parallel tasks and how CUDA lets us harness that power to speed up complex applications.

- **Real-World Uses of CUDA:** From signal processing to video editing and scientific simulations, we looked at how CUDA is making big impacts in various fields.

- **Understanding the Key Concepts:** We broke down essential ideas like the relationship between the Host (CPU) and Device (GPU), and how CUDA organizes work using threads, blocks, and grids.

- **The CUDA Programming Model:** We saw how CUDA makes it easy to manage thousands of threads without over-complicating things.

- **Setting Up Our CUDA Environment:** With help of other resurces, We followed a step-by-step guide for installing CUDA and making sure everything is ready to go on our system.

- **Writing Our First CUDA Program:** We coded a simple array addition program, which helped us grasp how to work with CUDA’s thread configuration, memory management, and kernel launches.

## Further Learning Resources:
- [CUDA by Example:](https://developer.nvidia.com/cuda-example) The best starting book for GPU Programming and beginner friendly.
- [CUDA Programming Guide:](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) A comprehensive guide with deeper insights into CUDA.
- [An Even Easier Introduction to CUDA:](https://developer.nvidia.com/blog/even-easier-introduction-cuda/) Practical beginner friendly resource with practical examples to enhance your skills.


