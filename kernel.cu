#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

// CUDA-ядро для поэлементного умножения векторов
__global__ void VecMulKernel(float *a, float *b, float *c) 
{ 
    // Определение индекса потока 
    int i = threadIdx.x + blockIdx.x * blockDim.x; 

    // Обработка соответствующей порции данных 
    c[i] = a[i] * b[i]; 
} 

// Функция-обертка для вызова ядра
void vec_mul_cuda(float *a, float *b, float *c, int n) 
{ 
    int SizeInBytes = n * sizeof(float); 

    // Указатели на массивы в видеопамяти 
    float *a_gpu = NULL; 
    float *b_gpu = NULL; 
    float *c_gpu = NULL; 

    // Выделение памяти под массивы на GPU 
    cudaMalloc((void **)&a_gpu, SizeInBytes); 
    cudaMalloc((void **)&b_gpu, SizeInBytes); 
    cudaMalloc((void **)&c_gpu, SizeInBytes); 

    // Копирование исходных данных из CPU на GPU 
    cudaMemcpy(a_gpu, a, SizeInBytes, cudaMemcpyHostToDevice); 
    cudaMemcpy(b_gpu, b, SizeInBytes, cudaMemcpyHostToDevice); 

    // Задание конфигурации запуска ядра 
    dim3 threads = dim3(512, 1);        // 512 потоков в блоке 
    dim3 blocks = dim3(n / threads.x, 1); // n/512 блоков в сетке 

    // Запуск ядра 
    VecMulKernel<<<blocks, threads>>>(a_gpu, b_gpu, c_gpu); 

    // Копирование результата из GPU в CPU 
    cudaMemcpy(c, c_gpu, SizeInBytes, cudaMemcpyDeviceToHost); 

    // Освобождение памяти GPU 
    cudaFree(a_gpu); 
    cudaFree(b_gpu); 
    cudaFree(c_gpu); 
}
