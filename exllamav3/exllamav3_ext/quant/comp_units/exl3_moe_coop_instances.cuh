#pragma once

// Per-K compilation units of the fused decode MoE kernels (exl3_moe_coop_kernel.cuh): each
// exl3_moe_coop_inst_k{K}.cu instantiates the gate/up (A) and down (B) kernels for the three
// codebooks. Returns the kernel pointer for cudaLaunchKernel and the dynamic shared memory size
// (A: for the given Hi; B: fixed)

struct MoeCoopKernel
{
    void* kernel;
    int smem;
};

// v: P3b kernel variant (0 = baseline, 3 = slice-ring kernels, 4 = fused persistent kernel). Variants 3 and 4 exist
// for the mul1 codebook only; other codebooks and unknown values use the baseline kernels
#define EXL3_MOE_COOP_INST_DECL(K) \
    MoeCoopKernel exl3_moe_coop_kernel_a_k##K(int cb, int Hi, bool wide, int v); \
    MoeCoopKernel exl3_moe_coop_kernel_b_k##K(int cb, bool wide, int v);

EXL3_MOE_COOP_INST_DECL(1)
EXL3_MOE_COOP_INST_DECL(2)
EXL3_MOE_COOP_INST_DECL(3)
EXL3_MOE_COOP_INST_DECL(4)
EXL3_MOE_COOP_INST_DECL(5)
EXL3_MOE_COOP_INST_DECL(6)
EXL3_MOE_COOP_INST_DECL(7)
EXL3_MOE_COOP_INST_DECL(8)

#undef EXL3_MOE_COOP_INST_DECL

// Half-integer rates K + 0.5 (exl3_moe_coop_inst_h{K}.cu), mul1 codebook only
#define EXL3_MOE_COOP_INST_DECL_H(K) \
    MoeCoopKernel exl3_moe_coop_kernel_a_h##K(int Hi, bool wide, int v); \
    MoeCoopKernel exl3_moe_coop_kernel_b_h##K(bool wide, int v);

EXL3_MOE_COOP_INST_DECL_H(1)
EXL3_MOE_COOP_INST_DECL_H(2)
EXL3_MOE_COOP_INST_DECL_H(3)

#undef EXL3_MOE_COOP_INST_DECL_H

// P3b variant 4: persistent fused gate/up + down kernel (mul1, gate/up and down at the same rate); {nullptr, 0}
// when the variant has no fused form
#define EXL3_MOE_COOP_INST_DECL_AB(N) MoeCoopKernel exl3_moe_coop_kernel_ab_##N(int Hi, bool wide_a, bool wide_b, int v);
EXL3_MOE_COOP_INST_DECL_AB(k1)
EXL3_MOE_COOP_INST_DECL_AB(k2)
EXL3_MOE_COOP_INST_DECL_AB(k3)
EXL3_MOE_COOP_INST_DECL_AB(k4)
EXL3_MOE_COOP_INST_DECL_AB(k5)
EXL3_MOE_COOP_INST_DECL_AB(k6)
EXL3_MOE_COOP_INST_DECL_AB(k7)
EXL3_MOE_COOP_INST_DECL_AB(k8)
EXL3_MOE_COOP_INST_DECL_AB(h1)
EXL3_MOE_COOP_INST_DECL_AB(h2)
EXL3_MOE_COOP_INST_DECL_AB(h3)
#undef EXL3_MOE_COOP_INST_DECL_AB

#define EXL3_MOE_COOP_INST_DEF_AB(N, K, H) \
    MoeCoopKernel exl3_moe_coop_kernel_ab_##N(int Hi, bool wide_a, bool wide_b, int v) \
    { \
        using namespace exl3_moe_coop_ns; \
        if (v != 4) return { nullptr, 0 }; \
        const int smem = smem_ab_bytes_v4<K, H>(Hi); \
        if (wide_a && wide_b) return { (void*) exl3_moe_coop_ab_kernel<K, 2, true, true, H, 4>, smem }; \
        if (wide_a) return { (void*) exl3_moe_coop_ab_kernel<K, 2, true, false, H, 4>, smem }; \
        if (wide_b) return { (void*) exl3_moe_coop_ab_kernel<K, 2, false, true, H, 4>, smem }; \
        return { (void*) exl3_moe_coop_ab_kernel<K, 2, false, false, H, 4>, smem }; \
    }

// P3b variants: kernel pointer and dynamic shared memory of variant v (mul1 codebook). Variant 4 takes the
// split variant 3 kernels for layers without a fused form (gate/up and down at different rates)
#define EXL3_MOE_COOP_VAR_A(K, H, W) \
    if (v == 3 || v == 4) return { (void*) exl3_moe_coop_a_kernel<K, 2, W, H, 3>, smem_a_bytes_ring<K, H>(Hi) };
#define EXL3_MOE_COOP_VAR_B(K, H, W) \
    if (v == 3 || v == 4) return { (void*) exl3_moe_coop_b_kernel<K, 2, W, H, 3>, smem_b_bytes_ring<K, H>() };

#define EXL3_MOE_COOP_INST_DEF_H(K) \
    MoeCoopKernel exl3_moe_coop_kernel_a_h##K(int Hi, bool wide, int v) \
    { \
        using namespace exl3_moe_coop_ns; \
        if (wide) { EXL3_MOE_COOP_VAR_A(K, true, true) } else { EXL3_MOE_COOP_VAR_A(K, true, false) } \
        const int smem = smem_a_bytes<K, true>(Hi); \
        if (wide) return { (void*) exl3_moe_coop_a_kernel<K, 2, true, true>, smem }; \
        return { (void*) exl3_moe_coop_a_kernel<K, 2, false, true>, smem }; \
    } \
    MoeCoopKernel exl3_moe_coop_kernel_b_h##K(bool wide, int v) \
    { \
        using namespace exl3_moe_coop_ns; \
        if (wide) { EXL3_MOE_COOP_VAR_B(K, true, true) } else { EXL3_MOE_COOP_VAR_B(K, true, false) } \
        const int smem = smem_b_bytes<K, true>(); \
        if (wide) return { (void*) exl3_moe_coop_b_kernel<K, 2, true, true>, smem }; \
        return { (void*) exl3_moe_coop_b_kernel<K, 2, false, true>, smem }; \
    }

// Body of an instance file
#define EXL3_MOE_COOP_INST_DEF(K) \
    MoeCoopKernel exl3_moe_coop_kernel_a_k##K(int cb, int Hi, bool wide, int v) \
    { \
        using namespace exl3_moe_coop_ns; \
        if (cb == 2) { if (wide) { EXL3_MOE_COOP_VAR_A(K, false, true) } else { EXL3_MOE_COOP_VAR_A(K, false, false) } } \
        const int smem = smem_a_bytes<K>(Hi); \
        if (wide) switch (cb) \
        { \
            case 0: return { (void*) exl3_moe_coop_a_kernel<K, 0, true>, smem }; \
            case 1: return { (void*) exl3_moe_coop_a_kernel<K, 1, true>, smem }; \
            default: return { (void*) exl3_moe_coop_a_kernel<K, 2, true>, smem }; \
        } \
        switch (cb) \
        { \
            case 0: return { (void*) exl3_moe_coop_a_kernel<K, 0, false>, smem }; \
            case 1: return { (void*) exl3_moe_coop_a_kernel<K, 1, false>, smem }; \
            default: return { (void*) exl3_moe_coop_a_kernel<K, 2, false>, smem }; \
        } \
    } \
    MoeCoopKernel exl3_moe_coop_kernel_b_k##K(int cb, bool wide, int v) \
    { \
        using namespace exl3_moe_coop_ns; \
        if (cb == 2) { if (wide) { EXL3_MOE_COOP_VAR_B(K, false, true) } else { EXL3_MOE_COOP_VAR_B(K, false, false) } } \
        const int smem = smem_b_bytes<K>(); \
        if (wide) switch (cb) \
        { \
            case 0: return { (void*) exl3_moe_coop_b_kernel<K, 0, true>, smem }; \
            case 1: return { (void*) exl3_moe_coop_b_kernel<K, 1, true>, smem }; \
            default: return { (void*) exl3_moe_coop_b_kernel<K, 2, true>, smem }; \
        } \
        switch (cb) \
        { \
            case 0: return { (void*) exl3_moe_coop_b_kernel<K, 0, false>, smem }; \
            case 1: return { (void*) exl3_moe_coop_b_kernel<K, 1, false>, smem }; \
            default: return { (void*) exl3_moe_coop_b_kernel<K, 2, false>, smem }; \
        } \
    }
