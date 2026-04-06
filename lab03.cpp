#include <cpuid.h>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>

struct Regs { uint32_t eax, ebx, ecx, edx; };

static Regs cpuid(uint32_t leaf, uint32_t subleaf = 0) {
    Regs r{};
    __cpuid_count(leaf, subleaf, r.eax, r.ebx, r.ecx, r.edx);
    return r;
}

static bool bit(uint32_t x, int b) { return (x >> b) & 1u; }
static uint32_t bits(uint32_t x, int lo, int hi) {
    return (x >> lo) & ((1u << (hi - lo + 1)) - 1u);
}

int main() {
    auto r0 = cpuid(0);
    uint32_t max_basic = r0.eax;

    char vendor[13];
    std::memcpy(vendor + 0, &r0.ebx, 4);
    std::memcpy(vendor + 4, &r0.edx, 4);
    std::memcpy(vendor + 8, &r0.ecx, 4);
    vendor[12] = '\0';

    std::cout << "Vendor" << vendor << "\n";
    std::cout << "Max basic leaf: 0x" << std::hex << max_basic << std::dec << "\n\n";

    auto rext0 = cpuid(0x80000000u);
    uint32_t max_ext = rext0.eax;
    std::cout << "Max extended leaf: 0x" << std::hex << max_ext << std::dec << "\n";

    if (max_ext >= 0x80000004u) {
        char brand[49]{};
        auto r2 = cpuid(0x80000002u);
        auto r3 = cpuid(0x80000003u);
        auto r4 = cpuid(0x80000004u);
        std::memcpy(brand +  0, &r2, 16);
        std::memcpy(brand + 16, &r3, 16);
        std::memcpy(brand + 32, &r4, 16);
        brand[48] = '\0';
        std::cout << "Brand: " << brand << "\n";
    }
    std::cout << "\n";

    if (max_basic >= 1) {
        auto r1 = cpuid(1);

        uint32_t stepping = bits(r1.eax, 0, 3);
        uint32_t model    = bits(r1.eax, 4, 7);
        uint32_t family   = bits(r1.eax, 8, 11);
        uint32_t ptype    = bits(r1.eax, 12, 13);
        uint32_t extModel = bits(r1.eax, 16, 19);
        uint32_t extFam   = bits(r1.eax, 20, 27);

        std::cout << "CPUID.1:EAX = 0x" << std::hex << r1.eax << std::dec << "\n";
        std::cout << "  Stepping=" << stepping
                  << " Model=" << model
                  << " Family=" << family
                  << " ProcType=" << ptype
                  << " ExtModel=" << extModel
                  << " ExtFamily=" << extFam << "\n";

        uint32_t max_logical = bits(r1.ebx, 16, 23);
        uint32_t apic_id     = bits(r1.ebx, 24, 31);
        std::cout << "  Max logical processors (pkg): " << max_logical << "\n";
        std::cout << "  Local APIC ID: " << apic_id << "\n";

        std::cout << "Features (EDX):"
                  << " FPU="  << bit(r1.edx, 0)
                  << " TSC="  << bit(r1.edx, 4)
                  << " MMX="  << bit(r1.edx, 23)
                  << " SSE="  << bit(r1.edx, 25)
                  << " SSE2=" << bit(r1.edx, 26)
                  << " HTT="  << bit(r1.edx, 28)
                  << "\n";

        std::cout << "Features (ECX):"
                  << " SSE3="   << bit(r1.ecx, 0)
                  << " SSSE3="  << bit(r1.ecx, 9)
                  << " FMA3="   << bit(r1.ecx, 12)
                  << " SSE4.1=" << bit(r1.ecx, 19)
                  << " SSE4.2=" << bit(r1.ecx, 20)
                  << " AVX="    << bit(r1.ecx, 28)
                  << "\n\n";
    }

    if (max_basic >= 4) {
        std::cout << "Caches (CPUID.4):\n";
        for (uint32_t i = 0; ; ++i) {
            auto rc = cpuid(4, i);
            if (rc.eax == 0) break;

            uint32_t cache_type  = bits(rc.eax, 0, 4);
            uint32_t cache_level = bits(rc.eax, 5, 7);
            uint32_t threads_sharing = bits(rc.eax, 14, 25) + 1;

            uint32_t line_size   = bits(rc.ebx, 0, 11) + 1;
            uint32_t partitions  = bits(rc.ebx, 12, 21) + 1;
            uint32_t ways        = bits(rc.ebx, 22, 31) + 1;
            uint32_t sets        = rc.ecx + 1;

            uint64_t size = uint64_t(line_size) * partitions * ways * uint64_t(sets);
            bool inclusive = bit(rc.edx, 1);

            const char* type_str = (cache_type==1)?"Data":(cache_type==2)?"Instruction":(cache_type==3)?"Unified":"Unknown";
            std::cout << "  subleaf " << i << ": " << type_str
                      << " L" << cache_level
                      << " shared_by=" << threads_sharing
                      << " line=" << line_size
                      << " ways=" << ways
                      << " sets=" << sets
                      << " inclusive=" << inclusive
                      << " size=" << (size/1024) << " KB\n";
        }
        std::cout << "\n";
    }

    if (max_basic >= 7) {
        auto r70 = cpuid(7, 0);
        uint32_t max_subleaf = r70.eax;
        std::cout << "CPUID.7.0: max subleaf = " << max_subleaf << "\n";
        std::cout << "  EBX:"
                  << " AVX2=" << bit(r70.ebx, 5)
                  << " RTM="  << bit(r70.ebx, 11)
                  << " AVX512F=" << bit(r70.ebx, 16)
                  << " SHA="  << bit(r70.ebx, 29)
                  << "\n";
        std::cout << "  ECX:"
                  << " GFNI=" << bit(r70.ecx, 8)
                  << "\n";
        std::cout << "  EDX: (AMX bits are here on some CPUs)\n";

        if (max_subleaf >= 1) {
            auto r71 = cpuid(7, 1);
            std::cout << "CPUID.7.1:\n";
            std::cout << "  EDX:"
                      << " (example) AVX10=" << bit(r71.edx, 19)
                      << "\n";
        }
        std::cout << "\n";
    }

    if (max_basic >= 0x16) {
        auto r16 = cpuid(0x16);
        uint32_t base = r16.eax & 0xFFFFu;
        uint32_t max  = r16.ebx & 0xFFFFu;
        uint32_t bus  = r16.ecx & 0xFFFFu;
        std::cout << "Frequencies (CPUID.16h, MHz): base=" << base
                  << " max=" << max << " bus=" << bus << "\n\n";
    }

    if (max_ext >= 0x80000001u) {
        auto r81 = cpuid(0x80000001u);
        std::cout << "Extended features (CPUID.80000001h):\n";
        std::cout << "  ECX: SSE4a=" << bit(r81.ecx, 6)
                  << " FMA4=" << bit(r81.ecx, 16) << "\n";
        std::cout << "  EDX: 3DNow=" << bit(r81.edx, 31)
                  << " Ext3DNow=" << bit(r81.edx, 30) << "\n";
    }

    return 0;
}
