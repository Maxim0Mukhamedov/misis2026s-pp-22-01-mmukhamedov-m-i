// method2_mmx_scalar.cpp
#include <immintrin.h>
#include <cstdint>
#include <cstddef>

// (2) С использованием указанного расширения ... со скалярными командами [1]
// Здесь "скалярность" = обработка только одного элемента массива за итерацию,
// но с использованием команд MMX: PMULLW + PACKSSWB [1].
extern "C" void scale_method2_mmx_scalar(int8_t* a, size_t n, int k) {
    __m64 vk = _mm_set1_pi16((short)k); // 4x int16

    for (size_t i = 0; i < n; ++i) {
        int16_t s = (int16_t)a[i]; // sign-extend byte -> int16

        // поместить s в младшее 16-битное слово MMX
        __m64 v = _mm_cvtsi32_si64((int)(uint16_t)s);

        // PMULLW: умножаем слова
        __m64 prod = _mm_mullo_pi16(v, vk);

        // PACKSSWB: int16 -> int8 с насыщением (signed saturate)
        __m64 packed = _mm_packs_pi16(prod, _mm_setzero_si64());

        int out32 = _mm_cvtsi64_si32(packed);
        a[i] = (int8_t)(out32 & 0xFF);
    }

    // После MMX нужно EMMS перед использованием x87/FPU [1]
    _mm_empty();
}
