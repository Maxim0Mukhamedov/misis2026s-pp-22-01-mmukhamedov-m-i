#include <iostream> 
#include <cuda.h> 
#include <cuda_runtime.h> 

using namespace std; 

// Описание внешней функции из .cu-файла 
void vec_mul_cuda(float *a, float *b, float *c, int n); 

const int N = 1024; // Размер вектора (кратен 512) 
float a[N], b[N], c[N]; 

int main() 
{ 
    // Заполнение векторов исходными данными 
    for (int i = 0; i < N; i++) 
    { 
        a[i] = (float)i; 
        b[i] = (float)i; 
        c[i] = 0; 
    } 

    // Покомпонентное умножение векторов на GPU 
    vec_mul_cuda(a, b, c, N); 

    // Вывод первых 20 значений результата 
    cout << "First 20 results: " << endl;
    for (int i = 0; i < 20; i++) 
      cout << c[i] << " "; 
    
    cout << endl;
    system("pause"); 
    return 0; 
}
