#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

// Подключение библиотеки CUDA Runtime
#pragma comment(lib, "cudart.lib")

int main() 
{ 
    int device_count; 
    cudaDeviceProp dp; 

    // Определение числа видеокарт с поддержкой CUDA
    cudaGetDeviceCount(&device_count); 
    cout << "CUDA device count: " << device_count << "\n"; 

    for (int i = 0; i < device_count; i++) 
    { 
        // Получение свойств i-го устройства
        cudaGetDeviceProperties(&dp, i); 
        
        cout << "\n--- Device " << i << ": " << dp.name << " ---\n";
        cout << "Compute Capability: " << dp.major << "." << dp.minor << "\n";
        cout << "Total Global Memory: " << dp.totalGlobalMem / (1024 * 1024) << " MB\n";
        cout << "Constant Memory Size: " << dp.totalConstMem << " bytes\n";
        cout << "Shared Memory per Block: " << dp.sharedMemPerBlock << " bytes\n";
        cout << "Registers per Block: " << dp.regsPerBlock << "\n";
        cout << "Warp Size: " << dp.warpSize << "\n";
        cout << "Max threads per block: " << dp.maxThreadsPerBlock << "\n";
        cout << "Multiprocessors (SM): " << dp.multiProcessorCount << "\n";
        cout << "Clock Rate: " << dp.clockRate << " kHz\n";
        cout << "Memory Clock Rate: " << dp.memoryClockRate << " kHz\n";
        cout << "L2 Cache Size: " << dp.l2CacheSize << " bytes\n";
        cout << "Memory Bus Width: " << dp.memoryBusWidth << " bits\n";
        cout << "Max Grid Dimensions: " << dp.maxGridSize[0] << " x " 
             << dp.maxGridSize[1] << " x " << dp.maxGridSize[2] << "\n";
        cout << "Max Block Dimensions: " << dp.maxThreadsPerBlock << " (total)\n";
    } 

    cout << "\nPress Enter to exit...";
    cin.get();
    return 0; 
}
