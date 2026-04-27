#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

void measure_bandwidth(const char* label, void* dst, void* src, size_t size, cudaMemcpyKind kind) {
    auto start = high_resolution_clock::now();
    
    cudaMemcpy(dst, src, size, kind);
    cudaThreadSynchronize(); // Ожидание завершения асинхронной операции [1]
    
    auto end = high_resolution_clock::now();
    duration<double> diff = end - start;
    
    double bandwidth = (size / (1024.0 * 1024.0 * 1024.0)) / diff.count(); 
    cout << label << " | Time: " << diff.count() << " s | Bandwidth: " << bandwidth << " GB/s" << endl;
}

int main() {
    size_t N = 1024 * 1024 * 256; // Размер блока данных (примерно 1 ГБ для float)
    size_t size = N * sizeof(float);

    float *h_src, *h_dst, *h_page_src, *h_page_dst;
    float *d_src, *d_dst;

    // 1. Выделение обычной памяти на Host
    h_src = (float*)malloc(size);
    h_dst = (float*)malloc(size);

    // 2. Выделение Page-locked памяти на Host [1]
    cudaMallocHost((void**)&h_page_src, size);
    cudaMallocHost((void**)&h_page_dst, size);

    // 3. Выделение памяти на Device
    cudaMalloc((void**)&d_src, size);
    cudaMalloc((void**)&d_dst, size);

    cout << "Measuring bandwidth for size: " << size / (1024*1024) << " MB\n" << endl;

    // Измерения
    measure_bandwidth("Host to Host", h_dst, h_src, size, cudaMemcpyHostToHost);
    measure_bandwidth("Host to Device (Normal)", d_src, h_src, size, cudaMemcpyHostToDevice);
    measure_bandwidth("Host to Device (Page-locked)", d_src, h_page_src, size, cudaMemcpyHostToDevice);
    measure_bandwidth("Device to Host (Normal)", h_dst, d_src, size, cudaMemcpyDeviceToHost);
    measure_bandwidth("Device to Host (Page-locked)", h_page_dst, d_src, size, cudaMemcpyDeviceToHost);
    measure_bandwidth("Device to Device", d_dst, d_src, size, cudaMemcpyDeviceToDevice);

    // Освобождение памяти
    free(h_src); free(h_dst);
    cudaFreeHost(h_page_src); cudaFreeHost(h_page_dst);
    cudaFree(d_src); cudaFree(d_dst);

    return 0;
}
