#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <cstring>
#include <cmath>
#include <chrono>
#include <iostream>

#include <emmintrin.h> // SSE2

struct Image {
    int w = 0, h = 0;
    unsigned char* data = nullptr;
};

static void freeImage(Image& img) {
    delete[] img.data;
    img.data = nullptr;
    img.w = img.h = 0;
}

static bool readToken(FILE* f, char* buf, int cap) {
    int c;
    // skip spaces and comments
    do {
        c = fgetc(f);
        if (c == '#') {
            while (c != '\n' && c != EOF) c = fgetc(f);
        }
    } while (std::isspace(c));

    if (c == EOF) return false;

    int i = 0;
    while (c != EOF && !std::isspace(c)) {
        if (i + 1 < cap) buf[i++] = (char)c;
        c = fgetc(f);
    }
    buf[i] = 0;
    return true;
}

static bool loadPGM(const char* path, Image& img) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return false;

    char tok[64];
    if (!readToken(f, tok, 64)) { std::fclose(f); return false; }
    if (std::strcmp(tok, "P5") != 0) { std::fclose(f); return false; }

    if (!readToken(f, tok, 64)) { std::fclose(f); return false; }
    img.w = std::atoi(tok);
    if (!readToken(f, tok, 64)) { std::fclose(f); return false; }
    img.h = std::atoi(tok);

    if (!readToken(f, tok, 64)) { std::fclose(f); return false; }
    int maxv = std::atoi(tok);
    if (maxv != 255) { std::fclose(f); return false; }

    // consume single whitespace after header (common)
    int c = fgetc(f);
    if (c == '\r') { int c2 = fgetc(f); if (c2 != '\n') ungetc(c2, f); }
    else if (c != '\n') { ungetc(c, f); }

    const size_t n = (size_t)img.w * (size_t)img.h;
    img.data = new unsigned char[n];
    size_t rd = std::fread(img.data, 1, n, f);
    std::fclose(f);

    if (rd != n) { freeImage(img); return false; }
    return true;
}

static bool savePGM(const char* path, const Image& img) {
    FILE* f = std::fopen(path, "wb");
    if (!f) return false;
    std::fprintf(f, "P5\n%d %d\n255\n", img.w, img.h);
    const size_t n = (size_t)img.w * (size_t)img.h;
    size_t wr = std::fwrite(img.data, 1, n, f);
    std::fclose(f);
    return wr == n;
}

static inline unsigned char clamp_u8(int x) {
    if (x < 0) return 0;
    if (x > 255) return 255;
    return (unsigned char)x;
}

// Roberts scalar:
// Hh = I(x,y) - I(x+1,y+1)
// Hv = I(x+1,y) - I(x,y+1)
// d  = floor(sqrt(Hh^2 + Hv^2))
static void robertsScalar(const Image& in, Image& out) {
    out.w = in.w;
    out.h = in.h;
    const size_t n = (size_t)in.w * (size_t)in.h;
    out.data = new unsigned char[n];
    std::memset(out.data, 0, n); // borders are 0

    const int w = in.w, h = in.h;
    const unsigned char* src = in.data;
    unsigned char* dst = out.data;

    for (int y = 0; y < h - 1; ++y) {
        const int row = y * w;
        const int rowNext = (y + 1) * w;
        for (int x = 0; x < w - 1; ++x) {
            int a = src[row + x];
            int b = src[row + x + 1];
            int c = src[rowNext + x];
            int d = src[rowNext + x + 1];

            int Hh = a - d;
            int Hv = b - c;

            int mag = (int)std::floor(std::sqrt((double)Hh * Hh + (double)Hv * Hv));
            dst[row + x] = clamp_u8(mag);
        }
    }
}

// SSE2 version: process 8 pixels at once (bytes -> words), compute Hh/Hv in 16-bit,
// then mag via scalar sqrt per lane (still much faster due to parallel diffs).
static void robertsSSE2(const Image& in, Image& out) {
    out.w = in.w;
    out.h = in.h;
    const size_t n = (size_t)in.w * (size_t)in.h;
    out.data = new unsigned char[n];
    std::memset(out.data, 0, n); // borders are 0

    const int w = in.w, h = in.h;
    const unsigned char* src = in.data;
    unsigned char* dst = out.data;

    const __m128i zero = _mm_setzero_si128();

    // We'll compute for x in [0..w-2]. Last column is border -> left 0.
    // SIMD block: 8 pixels (x..x+7), but must ensure x+7 <= w-2 => x <= w-9
    const int simdLimit = (w >= 2) ? (w - 1 - 8) : -1; // last x to start 8-wide safely

    // temp arrays for extracting 16-bit lanes
    alignas(16) short hh[8];
    alignas(16) short hv[8];

    for (int y = 0; y < h - 1; ++y) {
        const int row = y * w;
        const int rowNext = (y + 1) * w;

        int x = 0;
        for (; x <= simdLimit; x += 8) {
            // load 8 bytes from each needed position
            __m128i A8 = _mm_loadl_epi64((const __m128i*)(src + row + x));         // I(x..x+7,y)
            __m128i B8 = _mm_loadl_epi64((const __m128i*)(src + row + x + 1));     // I(x+1..x+8,y)
            __m128i C8 = _mm_loadl_epi64((const __m128i*)(src + rowNext + x));     // I(x..x+7,y+1)
            __m128i D8 = _mm_loadl_epi64((const __m128i*)(src + rowNext + x + 1)); // I(x+1..x+8,y+1)

            // unpack to 16-bit words
            __m128i A = _mm_unpacklo_epi8(A8, zero);
            __m128i B = _mm_unpacklo_epi8(B8, zero);
            __m128i C = _mm_unpacklo_epi8(C8, zero);
            __m128i D = _mm_unpacklo_epi8(D8, zero);

            __m128i Hh = _mm_sub_epi16(A, D);
            __m128i Hv = _mm_sub_epi16(B, C);

            _mm_store_si128((__m128i*)hh, Hh);
            _mm_store_si128((__m128i*)hv, Hv);

            // magnitude: sqrt(hh^2+hv^2) per lane (scalar here)
            for (int i = 0; i < 8; ++i) {
                int Hhi = (int)hh[i];
                int Hvi = (int)hv[i];
                int mag = (int)std::floor(std::sqrt((double)Hhi * Hhi + (double)Hvi * Hvi));
                dst[row + x + i] = clamp_u8(mag);
            }
        }

        // tail scalar for remaining x in [x..w-2]
        for (; x < w - 1; ++x) {
            int a = src[row + x];
            int b = src[row + x + 1];
            int c = src[rowNext + x];
            int d = src[rowNext + x + 1];

            int Hh = a - d;
            int Hv = b - c;
            int mag = (int)std::floor(std::sqrt((double)Hh * Hh + (double)Hv * Hv));
            dst[row + x] = clamp_u8(mag);
        }
    }
}

static size_t countDiff(const Image& a, const Image& b) {
    if (a.w != b.w || a.h != b.h) return (size_t)-1;
    const size_t n = (size_t)a.w * (size_t)a.h;
    size_t diff = 0;
    for (size_t i = 0; i < n; ++i)
        if (a.data[i] != b.data[i]) ++diff;
    return diff;
}

static double msSince(const std::chrono::high_resolution_clock::time_point& t0,
                      const std::chrono::high_resolution_clock::time_point& t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " input.pgm out_scalar.pgm out_simd.pgm\n";
        return 1;
    }

    Image in;
    if (!loadPGM(argv[1], in)) {
        std::cerr << "Failed to load PGM: " << argv[1] << "\n";
        return 1;
    }

    Image outScalar, outSIMD;

    // warmup / timing scalar
    auto t0 = std::chrono::high_resolution_clock::now();
    robertsScalar(in, outScalar);
    auto t1 = std::chrono::high_resolution_clock::now();

    // timing simd
    auto t2 = std::chrono::high_resolution_clock::now();
    robertsSSE2(in, outSIMD);
    auto t3 = std::chrono::high_resolution_clock::now();

    double msScalar = msSince(t0, t1);
    double msSIMD   = msSince(t2, t3);

    size_t diff = countDiff(outScalar, outSIMD);

    std::cerr << "Roberts scalar: " << msScalar << " ms\n";
    std::cerr << "Roberts SSE2  : " << msSIMD   << " ms\n";
    if (msSIMD > 0.0)
        std::cerr << "Speedup      : " << (msScalar / msSIMD) << "x\n";
    std::cerr << "Pixel diffs   : " << diff << "\n";

    if (!savePGM(argv[2], outScalar)) std::cerr << "Failed to save: " << argv[2] << "\n";
    if (!savePGM(argv[3], outSIMD))   std::cerr << "Failed to save: " << argv[3] << "\n";

    freeImage(in);
    freeImage(outScalar);
    freeImage(outSIMD);
    return 0;
}
