#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

// Функции из ваших трёх файлов:
extern "C" void scale_method1_cpp(int8_t* a, size_t n, int k);

extern "C" void scale_method2_mmx_scalar(int8_t* a, size_t n, int k);

extern "C" void scale_method3_vec_u1(int8_t* a, size_t n, int k);
extern "C" void scale_method3_vec_u2(int8_t* a, size_t n, int k);
extern "C" void scale_method3_vec_u4(int8_t* a, size_t n, int k);
extern "C" void scale_method3_vec_u8(int8_t* a, size_t n, int k);

// ---------------- helpers ----------------
static int8_t* aligned_alloc_16(size_t nbytes) {
    size_t nb = (nbytes + 15) & ~size_t(15);
    void* p = nullptr;
#if defined(_POSIX_C_SOURCE) && (_POSIX_C_SOURCE >= 200112L)
    if (posix_memalign(&p, 16, nb) != 0) return nullptr;
#else
    // fallback: может быть не выровнено, но обычно на x86-64 это не критично для данного кода
    p = std::malloc(nb);
    if (!p) return nullptr;
#endif
    return (int8_t*)p;
}

static void fill_data(int8_t* a, size_t n) {
    // детерминированный генератор для воспроизводимости
    uint32_t x = 123456789u;
    for (size_t i = 0; i < n; ++i) {
        x = 1664525u * x + 1013904223u;
        a[i] = (int8_t)(x >> 24);
    }
}

static void assert_equal_or_die(const int8_t* ref, const int8_t* got, size_t n, const char* name) {
    if (std::memcmp(ref, got, n) == 0) {
        std::printf("%s: OK\n", name);
        return;
    }
    std::printf("%s: MISMATCH\n", name);
    for (size_t i = 0; i < n; ++i) {
        if (ref[i] != got[i]) {
            std::printf("First diff at %zu: ref=%d got=%d\n", i, (int)ref[i], (int)got[i]);
            break;
        }
    }
    std::exit(2);
}

template <class Func>
static double bench_ms(Func f, int8_t* a, size_t n, int k, int iters) {
    // прогрев
    f(a, n, k);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) f(a, n, k);
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

// ---------------- main ----------------
int main(int argc, char** argv) {
    size_t n = 1'000'000; // пример из задания [1]
    int k = 3;
    int iters = 10;

    if (argc >= 2) n = (size_t)std::strtoull(argv[1], nullptr, 10);
    if (argc >= 3) k = std::atoi(argv[2]);
    if (argc >= 4) iters = std::atoi(argv[3]);

    int8_t* base = aligned_alloc_16(n);
    int8_t* ref  = aligned_alloc_16(n);
    int8_t* tmp  = aligned_alloc_16(n);

    if (!base || !ref || !tmp) {
        std::fprintf(stderr, "alloc failed\n");
        std::free(base); std::free(ref); std::free(tmp);
        return 1;
    }

    fill_data(base, n);

    // (1) Эталон — high-level C++ [1]
    std::memcpy(ref, base, n);
    scale_method1_cpp(ref, n, k);

    // Проверка совпадения для (2) и (3) [1]
    std::memcpy(tmp, base, n);
    scale_method2_mmx_scalar(tmp, n, k);
    assert_equal_or_die(ref, tmp, n, "2) MMX scalar");

    std::memcpy(tmp, base, n);
    scale_method3_vec_u1(tmp, n, k);
    assert_equal_or_die(ref, tmp, n, "3) MMX->SSE->MMX vector u1");

    std::memcpy(tmp, base, n);
    scale_method3_vec_u2(tmp, n, k);
    assert_equal_or_die(ref, tmp, n, "3) MMX->SSE->MMX vector u2");

    std::memcpy(tmp, base, n);
    scale_method3_vec_u4(tmp, n, k);
    assert_equal_or_die(ref, tmp, n, "3) MMX->SSE->MMX vector u4");

    std::memcpy(tmp, base, n);
    scale_method3_vec_u8(tmp, n, k);
    assert_equal_or_die(ref, tmp, n, "3) MMX->SSE->MMX vector u8");

    // Тайминги (каждый раз стартуем от одинакового base, чтобы было честно) [1]
    std::printf("\nTiming: n=%zu k=%d iters=%d\n", n, k, iters);

    std::memcpy(tmp, base, n);
    double t_cpp = bench_ms(scale_method1_cpp, tmp, n, k, iters);

    std::memcpy(tmp, base, n);
    double t_mmx_scalar = bench_ms(scale_method2_mmx_scalar, tmp, n, k, iters);

    std::memcpy(tmp, base, n);
    double t_u1 = bench_ms(scale_method3_vec_u1, tmp, n, k, iters);

    std::memcpy(tmp, base, n);
    double t_u2 = bench_ms(scale_method3_vec_u2, tmp, n, k, iters);

    std::memcpy(tmp, base, n);
    double t_u4 = bench_ms(scale_method3_vec_u4, tmp, n, k, iters);

    std::memcpy(tmp, base, n);
    double t_u8 = bench_ms(scale_method3_vec_u8, tmp, n, k, iters);

    std::printf("1) C++ scalar:                 %.3f ms\n", t_cpp);
    std::printf("2) MMX scalar (1 elem/iter):   %.3f ms\n", t_mmx_scalar);
    std::printf("3) MMX->SSE->MMX vector u1:    %.3f ms\n", t_u1);
    std::printf("3) MMX->SSE->MMX vector u2:    %.3f ms\n", t_u2);
    std::printf("3) MMX->SSE->MMX vector u4:    %.3f ms\n", t_u4);
    std::printf("3) MMX->SSE->MMX vector u8:    %.3f ms\n", t_u8);

    std::free(base);
    std::free(ref);
    std::free(tmp);
    return 0;
}
