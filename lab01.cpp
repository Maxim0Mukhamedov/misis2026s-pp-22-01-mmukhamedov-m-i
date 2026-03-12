#include <iostream>
#include <string>
#include <chrono>
#include <cstdint>
#include <mach/mach_time.h>
#include <fstream>
#include <cmath>
#include <iomanip>

static volatile std::uint64_t sink = 0;

std::uint64_t findCountNaive(const std::string& text, const std::string& pat) {
    if (pat.empty() || pat.size() > text.size()) return 0;
    std::uint64_t cnt = 0;
    for (size_t i = 0; i + pat.size() <= text.size(); ++i) {
        size_t j = 0;
        while (j < pat.size() && text[i + j] == pat[j]) ++j;
        if (j == pat.size()) ++cnt;
    }
    return cnt;
}


double mean(const double* a, int n) {
    long double s = 0;
    for (int i = 0; i < n; ++i) s += a[i];
    return (double)(s / n);
}


double varianceK(const double* a, int n, double avg) {
    long double s = 0;
    for (int i = 0; i < n; ++i) {
        long double d = (long double)a[i] - (long double)avg;
        s += d * d;
    }
    return (double)(s / n);
}

double minArr(const double* a, int n) {
    double mn = a[0];
    for (int i = 1; i < n; ++i) if (a[i] < mn) mn = a[i];
    return mn;
}


double confDelta95(double s, int n) {
    return 1.96 * s / std::sqrt((double)n);
}

void printStatsWithCI(const char* name, const double* times, int K) {
    double tmin = minArr(times, K);
    double avg = mean(times, K);
    double var = varianceK(times, K, avg);
    double s = std::sqrt(var);
    double delta = confDelta95(s, K);

    std::cout << name << ":\n";
    std::cout << "  min=" << tmin << " ms\n";
    std::cout << "  avg=" << avg << " ms\n";
    std::cout << "  s=" << s << " ms\n";
    std::cout << "  CI95: [" << (avg - delta) << "; " << (avg + delta) << "] ms, delta=" << delta << " ms\n";
}

void printAll(const double* times, int K) {
    for (int i = 0; i < K; ++i) {
        std::cout << times[i] << " ms";
        if (i + 1 < K) std::cout << ", ";
    }
    std::cout << "\n";
}


void writeArrayJson(std::ofstream& out, const char* key, const double* a, int n, bool commaAfter) {
    out << "  \"" << key << "\": [";
    for (int i = 0; i < n; ++i) {
        out << std::setprecision(17) << a[i];
        if (i + 1 < n) out << ", ";
    }
    out << "]";
    if (commaAfter) out << ",";
    out << "\n";
}

int main() {
    const int K = 5;
    const size_t N = 8'000'000;
    const std::string pat = "abcab";

    std::string text(N, 'a');
    for (size_t i = 0; i < N; ++i) text[i] = char('a' + (i % 26));
    for (size_t pos = 1000; pos + pat.size() < N; pos += 500000)
        text.replace(pos, pat.size(), pat);

    mach_timebase_info_data_t tb{};
    mach_timebase_info(&tb);

    sink = findCountNaive(text, pat);

    double sys_times[K];
    double st_times[K];
    double mach_times[K];

    for (int i = 0; i < K; ++i) {
        // 1) system_clock
        {
            auto s = std::chrono::system_clock::now();
            std::uint64_t c = findCountNaive(text, pat);
            auto e = std::chrono::system_clock::now();
            sink = c;

            sys_times[i] = std::chrono::duration<double, std::milli>(e - s).count();
        }

        // 2) steady_clock
        {
            auto s = std::chrono::steady_clock::now();
            std::uint64_t c = findCountNaive(text, pat);
            auto e = std::chrono::steady_clock::now();
            sink = c;

            st_times[i] = std::chrono::duration<double, std::milli>(e - s).count();
        }

        // 3) mach_absolute_time
        {
            uint64_t s = mach_absolute_time();
            std::uint64_t c = findCountNaive(text, pat);
            uint64_t e = mach_absolute_time();
            sink = c;

            uint64_t dt = e - s;
            long double ns = (long double)dt * (long double)tb.numer / (long double)tb.denom;
            mach_times[i] = (double)(ns / 1'000'000.0L);
        }
    }

    std::cout << "K=" << K << ", N=" << N << ", pat=\"" << pat << "\"\n\n";

    printStatsWithCI("system_clock", sys_times, K);
    std::cout << "  all_results: "; printAll(sys_times, K);
    std::cout << "\n";

    printStatsWithCI("steady_clock", st_times, K);
    std::cout << "  all_results: "; printAll(st_times, K);
    std::cout << "\n";

    printStatsWithCI("mach_time", mach_times, K);
    std::cout << "  all_results: "; printAll(mach_times, K);
    std::cout << "\n";

    std::cout << "sink=" << sink << "\n";

    // ---- JSON ----
    // Сохраняем только массивы времен + параметры
    std::ofstream out("times.json");
    out << "{\n";
    out << "  \"K\": " << K << ",\n";
    out << "  \"N\": " << N << ",\n";
    out << "  \"pattern\": \"" << pat << "\",\n";
    writeArrayJson(out, "system_clock_ms", sys_times, K, true);
    writeArrayJson(out, "steady_clock_ms", st_times, K, true);
    writeArrayJson(out, "mach_time_ms", mach_times, K, false);
    out << "}\n";
    out.close();

    return 0;
}
