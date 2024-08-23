# Chaptr 1: Introduction to CUDA and GPU Prallal Computing 
![CUDA Logo](https://github.com/user-attachments/assets/4ba2c60e-e699-4c4c-95e2-83499593f841)

## Overview

This guide will introduce you to:

- The basics of CUDA and why it’s significant
- Core concepts in CUDA programming
- How to write and run your first simple CUDA program

Let’s dive in!

## Why Learn CUDA?

In today's world, applications are increasingly data-heavy and complex. Traditional CPUs process tasks sequentially, which limits performance for parallel workloads. GPUs, with their thousands of cores, can execute many tasks simultaneously, making them ideal for high-performance applications like simulations, machine learning, and image processing. CUDA unlocks the full potential of GPU parallelism, enabling substantial acceleration in computing.

## Real-World Applications

- **Signal Processing and Communications:** Accelerate real-time tasks like radar processing, communication protocols, and sensor networks using CUDA to implement efficient algorithms such as Fast Fourier Transforms (FFT) and filtering.
  
- **Scientific Simulations:** Run complex models in fields like physics and molecular dynamics, where traditional CPUs struggle with large-scale, high-speed computations.

- **Embedded and Autonomous Systems:** Use CUDA in applications ranging from automotive to aerospace, enabling rapid prototyping and real-time processing for tasks like image analysis, sensor fusion, and control systems.

- **Advanced Data Analytics:** Analyze large datasets in research environments using CUDA to handle high-dimensional data efficiently, making processes like statistical modeling and experimental data analysis faster.

## Key Concepts in CUDA

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
   
   You’ll define the number of threads, blocks, and grids based on the problem you’re solving.

![CUDA Thread Hierarchy](https://github.com/user-attachments/assets/043ed75a-2540-4d43-8874-a50ff8e80128)

## The CUDA Programming Model

CUDA excels at managing thousands of concurrent threads while keeping code straightforward. It provides intuitive syntax for handling threads, memory, and execution—allowing you to focus on writing efficient algorithms.

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

1. Kernel Function (`add`):
2. 
The `__global__` keyword defines the add function as a CUDA kernel, meaning it runs directly on the GPU. Each thread operates on one element of the arrays, adding corresponding values from a and b and storing the result in c.

3. Memory Management:

Memory is allocated on both the host (CPU) and device (GPU) using cudaMalloc. The data is then copied from the host arrays (`h_a`, `h_b`) to the device arrays (`d_a`, `d_b`) using `cudaMemcpy`.

3. Grid and Block Configuration:

The kernel is launched with a configuration defined by dim3:
- `dim3 threadsPerBlock(n, 1, 1)`; sets n threads in a block (each thread handles one element).
- `dim3 blocksPerGrid(1, 1, 1)`; specifies a single block in the grid (since our data is small).
This setup ensures that all n elements are processed in parallel.

4. Kernel Launch:

The kernel is launched using `add<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c);`. This syntax specifies how the work is distributed across the GPU.

5. Copied Results and Cleanup:

After computation, the results are copied back to the host, displayed, and the device memory is freed using `cudaFree`.

## Output
This shows that the arrays are added correctly in parallel.

<img width="230" alt="image" src="https://github.com/user-attachments/assets/98f6adfa-75ab-46a6-b3ce-3806153f1f17">


## Further Learning Resources:
- [CUDA Programming Guide:](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) A comprehensive guide with deeper insights into CUDA.
- [An Even Easier Introduction to CUDA:](https://developer.nvidia.com/blog/even-easier-introduction-cuda/) Practical beginner friendly resource with practical examples to enhance your skills.


