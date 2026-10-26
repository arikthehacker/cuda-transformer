#pragma once
// mma.cuh: the two tensor core instructions Turing has, as inline PTX.
//
// mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32  (sm_75, PTX ISA 6.5)
//   d = a b + d, A 16x8 (row), B 8x8 (col), C/D 16x8 fp32. With g = lane / 4
//   and t = lane % 4, each lane holds:
//     a[0] = A[g][2t],   A[g][2t+1]        (two halves in one register)
//     a[1] = A[g+8][2t], A[g+8][2t+1]
//     b    = B[2t][g],   B[2t+1][g]
//     d[0], d[1] = C[g][2t],   C[g][2t+1]
//     d[2], d[3] = C[g+8][2t], C[g+8][2t+1]
//   Note that d's layout is a's layout: an fp32 result tile, packed to halves,
//   is already the A operand of the next mma (flash attention uses this).
//
// ldmatrix.sync.aligned.m8n8.x4{.trans}.shared.b16  (sm_75, PTX ISA 6.5)
//   Loads four 8x8 tiles of halves from shared memory. Lane 8j + r gives the
//   address of row r of tile j (16 bytes, 16-byte aligned). Register j of
//   lane l then holds, without .trans, elements 2(l%4) and 2(l%4)+1 of row l/4
//   of tile j (exactly an A or B fragment of a row-contiguous tile); with
//   .trans, element l/4 of rows 2(l%4) and 2(l%4)+1 (the same fragment, taken
//   from a tile stored the other way around).

#include <cstdint>
#include <cstring>
#include <cuda_fp16.h>

__device__ __forceinline__ uint32_t smem_u32(const void *p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t r[4], const void *smem_row)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem_u32(smem_row)));
}

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t r[4], const void *smem_row)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem_u32(smem_row)));
}

__device__ __forceinline__ void ldmatrix_x2(uint32_t r[2], const void *smem_row)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                 : "=r"(r[0]), "=r"(r[1])
                 : "r"(smem_u32(smem_row)));
}

__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t r[2], const void *smem_row)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                 : "=r"(r[0]), "=r"(r[1])
                 : "r"(smem_u32(smem_row)));
}

// d += a b on one 16 x 8 x 8 tile, fp16 in, fp32 accumulate
__device__ __forceinline__ void mma_16808(float d[4], const uint32_t a[2], uint32_t b)
{
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(b));
}

// two floats -> one register of two halves, lo in the low 16 bits
__device__ __forceinline__ uint32_t pack_half2(float lo, float hi)
{
    __half2 h = __floats2half2_rn(lo, hi);
    uint32_t r;
    memcpy(&r, &h, sizeof(r));              // a pointer cast here would be undefined (strict aliasing)
    return r;
}

// 16 bytes from global memory, or zeros when ok is false (p is not touched
// then, so it may point past the end). Written as one predicated PTX load:
// "v = ok ? *p : zero" in C++ can leave v in local memory instead of registers.
__device__ __forceinline__ int4 load_or_zero(const void *p, bool ok)
{
    int4 v;
    asm volatile("{\n"
                 "  .reg .pred q;\n"
                 "  setp.ne.b32 q, %5, 0;\n"
                 "  mov.b32 %0, 0;\n"
                 "  mov.b32 %1, 0;\n"
                 "  mov.b32 %2, 0;\n"
                 "  mov.b32 %3, 0;\n"
                 "  @q ld.global.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                 "}\n"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p), "r"(static_cast<int>(ok)));
    return v;
}
