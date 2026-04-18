// method1_cpp.cpp
#include <cstdint>
#include <cstddef>

static inline int8_t sat_i16_to_i8(int16_t x) {
    if (x > 127) return 127;
    if (x < -128) return -128;
    return (int8_t)x;
}

// (1) На языке высокого уровня стандартными средствами работы с массивами [1]
extern "C" void scale_method1_cpp(int8_t* a, size_t n, int k) {
    for (size_t i = 0; i < n; ++i) {
        int16_t prod = (int16_t)a[i] * (int16_t)k; // 8-bit * 8-bit => 16-bit [1]
        a[i] = sat_i16_to_i8(prod);
    }
}
