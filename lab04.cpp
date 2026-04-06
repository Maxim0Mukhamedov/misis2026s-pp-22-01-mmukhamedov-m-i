#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <chrono>

static void progress_bar(const char* title, int cur, int total) {
    const int width = 40;
    int filled = (int)((int64_t)cur * width / total);

    std::printf("\r%s [", title);
    for (int i = 0; i < width; i++) std::printf(i < filled ? "#" : " ");
    int percent = (int)((int64_t)cur * 100 / total);
    std::printf("] %3d%% (%d/%d)", percent, cur, total);
    std::fflush(stdout);

    if (cur == total) std::printf("\n");
}

// Простой LCG, чтобы не тащить сложные генераторы и не тратить много времени
static inline uint32_t lcg_next(uint32_t &state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

static double now_ns() {
    using namespace std::chrono;
    return (double)duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count();
}

static void fill_array_int(int* a, int n) {
    for (int i = 0; i < n; i++) a[i] = (i & 1023);
}

// Последовательный обход
static double bench_sequential(const int* a, int n, int repeats) {
    volatile int64_t sink = 0;
    double t0 = now_ns();
    for (int r = 0; r < repeats; r++) {
        int64_t s = 0;
        for (int i = 0; i < n; i++) s += a[i];
        sink += s;
    }
    double t1 = now_ns();
    (void)sink;
    // время одной итерации внутреннего цикла (одного доступа/сложения)
    return (t1 - t0) / (double)(repeats * (int64_t)n);
}

// Случайный индекс каждый раз (генерация индекса внутри цикла)
static double bench_random_inline(const int* a, int n, int repeats) {
    volatile int64_t sink = 0;
    uint32_t st = 123456789u;

    double t0 = now_ns();
    for (int r = 0; r < repeats; r++) {
        int64_t s = 0;
        for (int i = 0; i < n; i++) {
            uint32_t x = lcg_next(st);
            int idx = (int)(x % (uint32_t)n);
            s += a[idx];
        }
        sink += s;
    }
    double t1 = now_ns();
    (void)sink;
    return (t1 - t0) / (double)(repeats * (int64_t)n);
}

// Случайный обход по заранее подготовленному массиву индексов
static double bench_random_index_array(const int* a, const int* idx_arr, int n, int repeats) {
    volatile int64_t sink = 0;

    double t0 = now_ns();
    for (int r = 0; r < repeats; r++) {
        int64_t s = 0;
        for (int i = 0; i < n; i++) {
            s += a[idx_arr[i]];
        }
        sink += s;
    }
    double t1 = now_ns();
    (void)sink;
    return (t1 - t0) / (double)(repeats * (int64_t)n);
}

static void build_index_array(int* idx_arr, int n) {
    uint32_t st = 987654321u;
    for (int i = 0; i < n; i++) {
        uint32_t x = lcg_next(st);
        idx_arr[i] = (int)(x % (uint32_t)n);
    }
}

static int repeats_for_size_bytes(int bytes) {
    // Чтобы замеры не были слишком долгими, но были стабильными:
    // маленькие размеры прогоняем больше раз, большие — меньше.
    if (bytes <= 256 * 1024) return 400;
    if (bytes <= 2 * 1024 * 1024) return 200;
    if (bytes <= 32 * 1024 * 1024) return 80;
    return 30; // до ~150 МБ
}

int main() {
    // Диапазоны как в методичке (рекомендации) [1]
    // 1) до 2 МБ шаг 1 КБ
    // 2) до 32 МБ шаг 512 КБ
    // 3) до 150 МБ шаг 5 МБ
    const int max_mb = 150;
    const int max_bytes = max_mb * 1024 * 1024;

    std::FILE* f = std::fopen("results.csv", "wb");
    if (!f) {
        std::printf("Не могу создать results.csv\n");
        return 1;
    }
    std::fprintf(f, "method,bytes,n,ns_per_iter\n");

    // Чтобы не выделять память много раз: выделим максимум один раз.
    int max_n = max_bytes / (int)sizeof(int);
    int* a = (int*)std::malloc((size_t)max_n * sizeof(int));
    int* idx_arr = (int*)std::malloc((size_t)max_n * sizeof(int));
    if (!a || !idx_arr) {
        std::printf("Не хватило памяти для массивов\n");
        return 1;
    }

    fill_array_int(a, max_n);

    // Соберём список размеров в байтах простыми циклами
    const int cap = 10000;
    int* sizes = (int*)std::malloc(cap * sizeof(int));
    int count = 0;

    // 1) 1KB..2MB шаг 1KB
    for (int b = 1 * 1024; b <= 2 * 1024 * 1024; b += 1 * 1024) sizes[count++] = b;
    // 2) 2.5MB..32MB шаг 512KB (чтобы не дублировать 2MB)
    for (int b = 2 * 1024 * 1024 + 512 * 1024; b <= 32 * 1024 * 1024; b += 512 * 1024) sizes[count++] = b;
    // 3) 37MB..150MB шаг 5MB
    for (int b = 32 * 1024 * 1024 + 5 * 1024 * 1024; b <= 150 * 1024 * 1024; b += 5 * 1024 * 1024) sizes[count++] = b;

    if (count > cap) {
        std::printf("Слишком много размеров, увеличьте cap\n");
        return 1;
    }

    // --- Метод 1: последовательный ---
    for (int i = 0; i < count; i++) {
        int bytes = sizes[i];
        int n = bytes / (int)sizeof(int);
        if (n <= 0) continue;

        int repeats = repeats_for_size_bytes(bytes);
        double ns_iter = bench_sequential(a, n, repeats);

        std::fprintf(f, "sequential,%d,%d,%.3f\n", bytes, n, ns_iter);
        progress_bar("Метод 1/3: последовательный", i + 1, count);
    }

    // --- Метод 2: случайный (индекс генерируется внутри) ---
    for (int i = 0; i < count; i++) {
        int bytes = sizes[i];
        int n = bytes / (int)sizeof(int);
        if (n <= 0) continue;

        int repeats = repeats_for_size_bytes(bytes);
        double ns_iter = bench_random_inline(a, n, repeats);

        std::fprintf(f, "random_inline,%d,%d,%.3f\n", bytes, n, ns_iter);
        progress_bar("Метод 2/3: случайный inline", i + 1, count);
    }

    // --- Метод 3: случайный по массиву индексов ---
    // Важно: для каждого размера пересоздаём idx_arr[0..n-1], чтобы он соответствовал n.
    for (int i = 0; i < count; i++) {
        int bytes = sizes[i];
        int n = bytes / (int)sizeof(int);
        if (n <= 0) continue;

        build_index_array(idx_arr, n);

        int repeats = repeats_for_size_bytes(bytes);
        double ns_iter = bench_random_index_array(a, idx_arr, n, repeats);

        std::fprintf(f, "random_index_array,%d,%d,%.3f\n", bytes, n, ns_iter);
        progress_bar("Метод 3/3: случайный index_arr", i + 1, count);
    }

    std::fclose(f);
    std::free(a);
    std::free(idx_arr);
    std::free(sizes);

    std::printf("Готово. Результаты: results.csv\n");
    return 0;
}
