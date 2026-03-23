#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <chrono>

static inline double now_sec() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

static inline double gflops(int N, double tsec) {
    // ~ N^3 умножений + N^3 сложений => 2*N^3 FLOP [1]
    double flops = 2.0 * (double)N * (double)N * (double)N;
    return flops / tsec / 1e9;
}

static float* alloc_mat(int N) {
    size_t bytes = (size_t)N * (size_t)N * sizeof(float);
    float* p = (float*)std::malloc(bytes);
    if (!p) {
        std::fprintf(stderr, "alloc failed (%zu bytes)\n", bytes);
        std::exit(1);
    }
    return p;
}

static void zero_mat(float* C, int N) {
    std::memset(C, 0, (size_t)N*(size_t)N*sizeof(float));
}

static void init_mat(float* A, int N, uint32_t seed) {
    // детерминированная "псевдо-рандомная" инициализация без <random>
    // значения примерно в [-1;1]
    for (int i=0;i<N;i++) {
        for (int j=0;j<N;j++) {
            uint32_t x = seed ^ (uint32_t)(i*1664525u + j*1013904223u);
            x = x * 1664525u + 1013904223u;
            float v = (float)((int)(x & 0xFFFF) - 32768) / 32768.0f;
            A[(size_t)i*N + j] = v;
        }
    }
}

static bool compare_mat(const float* X, const float* Y, int N, float rtol=1e-3f, float atol=1e-3f) {
    size_t nn = (size_t)N*(size_t)N;
    for (size_t idx=0; idx<nn; ++idx) {
        float x = X[idx], y = Y[idx];
        float diff = std::fabs(x - y);
        float tol = atol + rtol * (std::fmax(std::fabs(x), std::fabs(y)));
        if (diff > tol) return false;
    }
    return true;
}

// ---------------- 1) classic ijk [1] ----------------
static void mul_classic(const float* A, const float* B, float* C, int N) {
    zero_mat(C, N);
    for (int i=0;i<N;i++) {
        for (int j=0;j<N;j++) {
            float s = 0.0f;
            const float* arow = A + (size_t)i*N;
            for (int k=0;k<N;k++) {
                s += arow[k] * B[(size_t)k*N + j];
            }
            C[(size_t)i*N + j] = s;
        }
    }
}

// -------------- 2) transpose B then multiply [1] --------------
static void transpose(const float* B, float* BT, int N) {
    for (int i=0;i<N;i++)
        for (int j=0;j<N;j++)
            BT[(size_t)j*N + i] = B[(size_t)i*N + j];
}

static void mul_with_BT(const float* A, const float* BT, float* C, int N) {
    zero_mat(C, N);
    for (int i=0;i<N;i++) {
        const float* arow = A + (size_t)i*N;
        for (int j=0;j<N;j++) {
            const float* btrow = BT + (size_t)j*N;
            float s = 0.0f;
            for (int k=0;k<N;k++) s += arow[k] * btrow[k];
            C[(size_t)i*N + j] = s;
        }
    }
}

// -------------- 3) buffer column B, order j,i,k [1] + unroll M [1] --------------
template<int M>
static void mul_buffer_colB(const float* A, const float* B, float* C, int N, float* tmp) {
    zero_mat(C, N);
    for (int j=0;j<N;j++) {
        for (int k=0;k<N;k++) tmp[k] = B[(size_t)k*N + j];

        for (int i=0;i<N;i++) {
            const float* arow = A + (size_t)i*N;
            float acc[M] = {0};

            int k=0;
            for (; k + M <= N; k += M) {
                #pragma clang loop unroll(disable)
                for (int u=0; u<M; ++u) acc[u] += arow[k+u] * tmp[k+u];
            }
            float s = 0.0f;
            for (int u=0; u<M; ++u) s += acc[u];
            for (; k<N; ++k) s += arow[k] * tmp[k];

            C[(size_t)i*N + j] = s;
        }
    }
}

// -------------- 4) blocked SxS + on-the-fly transpose of B-block [1] + unroll M [1] --------------
template<int M>
static void mul_blocked(const float* A, const float* B, float* C, int N, int S,
                        float* Abuf, float* BbufT) {
    zero_mat(C, N);

    for (int ii=0; ii<N; ii+=S)
    for (int jj=0; jj<N; jj+=S)
    for (int kk=0; kk<N; kk+=S) {
        int iimax = (ii+S < N) ? (ii+S) : N;
        int jjmax = (jj+S < N) ? (jj+S) : N;
        int kkmax = (kk+S < N) ? (kk+S) : N;

        int Si = iimax - ii;
        int Sj = jjmax - jj;
        int Sk = kkmax - kk;

        // copy A-block (ii.., kk..) into Abuf (row-major, stride S)
        for (int i=0;i<Si;i++)
            for (int k=0;k<Sk;k++)
                Abuf[(size_t)i*S + k] = A[(size_t)(ii+i)*N + (kk+k)];

        // copy B-block (kk.., jj..) into BbufT transposed "on the fly" [1]
        for (int k=0;k<Sk;k++)
            for (int j=0;j<Sj;j++)
                BbufT[(size_t)j*S + k] = B[(size_t)(kk+k)*N + (jj+j)];

        // update C-block
        for (int i=0;i<Si;i++)
        for (int j=0;j<Sj;j++) {
            const float* ap = Abuf + (size_t)i*S;
            const float* bp = BbufT + (size_t)j*S;

            float acc[M] = {0};
            int k=0;
            for (; k + M <= Sk; k += M) {
                #pragma clang loop unroll(disable)
                for (int u=0; u<M; ++u) acc[u] += ap[k+u] * bp[k+u];
            }
            float s = 0.0f;
            for (int u=0; u<M; ++u) s += acc[u];
            for (; k<Sk; ++k) s += ap[k] * bp[k];

            C[(size_t)(ii+i)*N + (jj+j)] += s;
        }
    }
}

static void csv_header(FILE* f) {
    std::fprintf(f, "alg,N,S,M,t_sec,gflops,t_transpose_sec,gflops_no_transpose\n");
}

static void csv_row(FILE* f, const char* alg, int N, int S, int M,
                    double t, double gf, double tT, double gfNoT) {
    std::fprintf(f, "%s,%d,%d,%d,%.9f,%.6f,%.9f,%.6f\n",
                 alg, N, S, M, t, gf, tT, gfNoT);
}

int main() {
    // N можно расширять до 8192 (учтите RAM/время) [1]
    const int Ns[] = {256, 512, 1024};
    const int numN = (int)(sizeof(Ns)/sizeof(Ns[0]));

    // S степенями двойки: 1,2,4,8,... [1]
    const int Ss[] = {1,2,4,8,16,32,64,128};
    const int numS = (int)(sizeof(Ss)/sizeof(Ss[0]));

    const int Ms[] = {1,2,4,8,16}; // раскрутка M степенями двойки [1]
    const int numM = (int)(sizeof(Ms)/sizeof(Ms[0]));

    const int repeats = 3;

    FILE* f = std::fopen("results.csv", "wb");
    if (!f) { std::perror("results.csv"); return 1; }
    csv_header(f);

    for (int ni=0; ni<numN; ++ni) {
        int N = Ns[ni];

        float* A  = alloc_mat(N);
        float* B  = alloc_mat(N);
        float* C0 = alloc_mat(N); // classic reference
        float* C1 = alloc_mat(N);

        init_mat(A, N, 111);
        init_mat(B, N, 222);

        // 1) classic
        {
            double best = 1e100;
            for (int r=0;r<repeats;r++) {
                double t0 = now_sec();
                mul_classic(A,B,C0,N);
                double t = now_sec() - t0;
                if (t < best) best = t;
            }
            csv_row(f, "classic_ijk", N, 0, 1, best, gflops(N,best), 0.0, 0.0);
        }

        // 2) transpose B variant: with and without transpose time [1]
        {
            float* BT = alloc_mat(N);

            double bestT = 1e100;
            for (int r=0;r<repeats;r++) {
                double t0 = now_sec();
                transpose(B,BT,N);
                double t = now_sec() - t0;
                if (t < bestT) bestT = t;
            }

            double bestMul = 1e100;
            for (int r=0;r<repeats;r++) {
                double t0 = now_sec();
                mul_with_BT(A,BT,C1,N);
                double t = now_sec() - t0;
                if (t < bestMul) bestMul = t;
            }

            if (!compare_mat(C0, C1, N)) {
                std::fprintf(stderr, "Mismatch transposeB at N=%d\n", N);
                return 2;
            }

            double t_total = bestT + bestMul;
            csv_row(f, "transposeB_ijk", N, 0, 1,
                    t_total, gflops(N,t_total),
                    bestT, gflops(N,bestMul));

            std::free(BT);
        }

        // 3) buffered column B + unroll M [1]
        {
            float* tmp = (float*)std::malloc((size_t)N*sizeof(float));
            if (!tmp) { std::fprintf(stderr, "tmp alloc failed\n"); return 1; }

            for (int mi=0; mi<numM; ++mi) {
                int M = Ms[mi];
                double best = 1e100;

                for (int r=0;r<repeats;r++) {
                    double t0 = now_sec();
                    if (M==1)  mul_buffer_colB<1>(A,B,C1,N,tmp);
                    if (M==2)  mul_buffer_colB<2>(A,B,C1,N,tmp);
                    if (M==4)  mul_buffer_colB<4>(A,B,C1,N,tmp);
                    if (M==8)  mul_buffer_colB<8>(A,B,C1,N,tmp);
                    if (M==16) mul_buffer_colB<16>(A,B,C1,N,tmp);
                    double t = now_sec() - t0;
                    if (t < best) best = t;
                }

                if (!compare_mat(C0, C1, N)) {
                    std::fprintf(stderr, "Mismatch buffer_colB M=%d at N=%d\n", M, N);
                    return 3;
                }
                csv_row(f, "buffer_colB", N, 0, M, best, gflops(N,best), 0.0, 0.0);
            }

            std::free(tmp);
        }

        // 4) blocked: sweep S and M, block sizes are powers of 2 [1]
        for (int si=0; si<numS; ++si) {
            int S = Ss[si];
            if (S > N) continue;

            float* Abuf = (float*)std::malloc((size_t)S*(size_t)S*sizeof(float));
            float* BbufT= (float*)std::malloc((size_t)S*(size_t)S*sizeof(float));
            if (!Abuf || !BbufT) { std::fprintf(stderr, "block buf alloc failed\n"); return 1; }

            for (int mi=0; mi<numM; ++mi) {
                int M = Ms[mi];
                double best = 1e100;

                for (int r=0;r<repeats;r++) {
                    double t0 = now_sec();
                    if (M==1)  mul_blocked<1>(A,B,C1,N,S,Abuf,BbufT);
                    if (M==2)  mul_blocked<2>(A,B,C1,N,S,Abuf,BbufT);
                    if (M==4)  mul_blocked<4>(A,B,C1,N,S,Abuf,BbufT);
                    if (M==8)  mul_blocked<8>(A,B,C1,N,S,Abuf,BbufT);
                    if (M==16) mul_blocked<16>(A,B,C1,N,S,Abuf,BbufT);
                    double t = now_sec() - t0;
                    if (t < best) best = t;
                }

                if (!compare_mat(C0, C1, N)) {
                    std::fprintf(stderr, "Mismatch blocked S=%d M=%d at N=%d\n", S, M, N);
                    return 4;
                }
                csv_row(f, "blocked", N, S, M, best, gflops(N,best), 0.0, 0.0);
            }

            std::free(Abuf);
            std::free(BbufT);
        }

        std::free(A);
        std::free(B);
        std::free(C0);
        std::free(C1);

        std::fprintf(stderr, "Done N=%d\n", N);
    }

    std::fclose(f);
    std::fprintf(stderr, "Wrote results.csv\n");
    return 0;
}
