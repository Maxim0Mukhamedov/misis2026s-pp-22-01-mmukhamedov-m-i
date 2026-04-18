// method3_mmx_sse_mmx_vector.cpp
#include <immintrin.h>
#include <cstdint>
#include <cstddef>

static inline int8_t sat_i16_to_i8(int16_t x) {
    if (x > 127) return 127;
    if (x < -128) return -128;
    return (int8_t)x;
}

static inline void scale_block8_mmx_sse_mmx(int8_t* p, __m128i vk16, __m128i vzero) {
    __m64 mm = *reinterpret_cast<const __m64*>(p);

    // MMX -> SSE
    __m128i x = _mm_movpi64_epi64(mm);

    // sign-extend int8 -> int16
    __m128i sign = _mm_cmpgt_epi8(vzero, x);
    __m128i w16  = _mm_unpacklo_epi8(x, sign);   // 8x int16

    // multiply
    __m128i prod16 = _mm_mullo_epi16(w16, vk16);

    // pack int16 -> int8 with signed saturation
    __m128i packed = _mm_packs_epi16(prod16, prod16);

    // SSE -> MMX store low 64
    long long low64 = _mm_cvtsi128_si64(packed);
    __m64 out = _mm_cvtsi64_m64(low64);

    *reinterpret_cast<__m64*>(p) = out;
}

// (3) Векторные команды указанного расширения; горизонтальные — при необходимости [1].
// Здесь: MMX→SSE→MMX, обработка по 8 int8 за итерацию + раскрутка 2/4/8 [1].

extern "C" void scale_method3_vec_u1(int8_t* a, size_t n, int k) {
    __m128i vk16  = _mm_set1_epi16((short)k);
    __m128i vzero = _mm_setzero_si128();

    size_t i = 0;
    for (; i + 8 <= n; i += 8) scale_block8_mmx_sse_mmx(a + i, vk16, vzero);

    for (; i < n; ++i) {
        int16_t prod = (int16_t)a[i] * (int16_t)k;
        a[i] = sat_i16_to_i8(prod);
    }
    _mm_empty();
}

extern "C" void scale_method3_vec_u2(int8_t* a, size_t n, int k) {
    __m128i vk16  = _mm_set1_epi16((short)k);
    __m128i vzero = _mm_setzero_si128();

    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        scale_block8_mmx_sse_mmx(a + i + 0, vk16, vzero);
        scale_block8_mmx_sse_mmx(a + i + 8, vk16, vzero);
    }
    for (; i + 8 <= n; i += 8) scale_block8_mmx_sse_mmx(a + i, vk16, vzero);

    for (; i < n; ++i) {
        int16_t prod = (int16_t)a[i] * (int16_t)k;
        a[i] = sat_i16_to_i8(prod);
    }
    _mm_empty();
}

extern "C" void scale_method3_vec_u4(int8_t* a, size_t n, int k) {
    __m128i vk16  = _mm_set1_epi16((short)k);
    __m128i vzero = _mm_setzero_si128();

    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        scale_block8_mmx_sse_mmx(a + i +  0, vk16, vzero);
        scale_block8_mmx_sse_mmx(a + i +  8, vk16, vzero);
        scale_block8_mmx_sse_mmx(a + i + 16, vk16, vzero);
        scale_block8_mmx_sse_mmx(a + i + 24, vk16, vzero);
    }
    for (; i + 8 <= n; i += 8) scale_block8_mmx_sse_mmx(a + i, vk16, vzero);

    for (; i < n; ++i) {
        int16_t prod = (int16_t)a[i] * (int16_t)k;
        a[i] = sat_i16_to_i8(prod);
    }
    _mm_empty();
}

extern "C" void scale_method3_vec_u8(int8_t* a, size_t n, int k) {
    __m128i vk16  = _mm_set1_epi16((short)k);
    __m128i vzero = _mm_setzero_si128();

    size_t i = 0;
    for (; i + 64 <= n; i += 64) {
        scale_block8_mmx_sse_mmx(a + i +  0, vk16, vzero);
        scale_block8_mmx_sse_mmx(a + i +  8, vk16, vzero);
        scale_block8_mmx_sse_mmx(a + i + 16, vk16, vzero);
        scale_block8_mmx_sse_mmx(a + i + 24, vk16, vzero);
        scale_block8_mmx_sse_mmx(a + i + 32, vk16, vzero);
        scale_block8_mmx_sse_mmx(a + i + 40, vk16, vzero);
        scale_block8_mmx_sse_mmx(a + i + 48, vk16, vzero);
        scale_block8_mmx_sse_mmx(a + i + 56, vk16, vzero);
    }
    for (; i + 8 <= n; i += 8) scale_block8_mmx_sse_mmx(a + i, vk16, vzero);

    for (; i < n; ++i) {
        int16_t prod = (int16_t)a[i] * (int16_t)k;
        a[i] = sat_i16_to_i8(prod);
    }
    _mm_empty();
}
