#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>

// Cubiomes 라이브러리 (C 언어) 연결
extern "C" {
    #include "finders.h"
    #include "generator.h"
    #include "biomes.h"
}

// =========================================================
// [USER INPUT] 좌표 입력 구간
// =========================================================
const int MC_VERSION = MC_1_16_1;

const int BT1_X = 1234; 
const int BT1_Z = -5678;

const int BT2_X = 4321; 
const int BT2_Z = -8765;

const int SH_X = -1500;
const int SH_Z = 300;

// RTX 3070 Ti 최적화 설정
#define THREADS_PER_BLOCK 256
#define BLOCKS_PER_GRID 65536 
#define BATCH_SIZE (THREADS_PER_BLOCK * BLOCKS_PER_GRID)

// =========================================================
// [GPU KERNEL] 충돌 방지를 위해 함수명 앞에 'gpu_' 추가
// =========================================================

__device__ __forceinline__ uint64_t gpu_nextSeed(uint64_t *seed) {
    *seed = (*seed * 0x5DEECE66DULL + 0xBULL) & ((1ULL << 48) - 1);
    return *seed;
}

__device__ __forceinline__ int gpu_nextInt(uint64_t *seed, int n) {
    uint64_t bits, val;
    // n is power of 2 check
    if ((n & -n) == n) { 
        *seed = (*seed * 0x5DEECE66DULL + 0xBULL) & ((1ULL << 48) - 1);
        return (int)((n * (*seed >> 17)) >> 31);
    }
    
    // 일반적인 경우
    do {
        *seed = (*seed * 0x5DEECE66DULL + 0xBULL) & ((1ULL << 48) - 1);
        bits = *seed >> 17;
        val = bits % n;
    } while ((int64_t)(bits - val + (n - 1)) < 0); // signed comparison fix
    return val;
}

__device__ __forceinline__ float gpu_nextFloat(uint64_t *seed) {
    *seed = (*seed * 0x5DEECE66DULL + 0xBULL) & ((1ULL << 48) - 1);
    return (int)(*seed >> 24) / ((float)(1 << 24));
}

// 1.13+ Buried Treasure 존재 여부 확인
__device__ bool checkTreasure(uint64_t lower48, int chunkX, int chunkZ) {
    // 1. 시드 초기화 (Structure Seed + Salt + Region Coordinates)
    // Buried Treasure Salt = 10387320
    uint64_t seed = lower48 + 10387320ULL;
    
    // Chunk 좌표 해시
    seed += (uint64_t)chunkX * 341873128712ULL + (uint64_t)chunkZ * 132897987541ULL;
    
    // Java Random setSeed
    seed = (seed ^ 0x5DEECE66DULL) & ((1ULL << 48) - 1);

    // 2. 확률 체크 (1%)
    // gpu_nextFloat 사용
    return gpu_nextFloat(&seed) < 0.01f;
}

// GPU 커널
__global__ void search_kernel(uint64_t start_offset, int bt1_cx, int bt1_cz, int bt2_cx, int bt2_cz, 
                              bool use_bt2, uint64_t *results, int *result_count, int max_results) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t current_seed = start_offset + idx;

    // 1차 필터: BT1
    if (checkTreasure(current_seed, bt1_cx, bt1_cz)) {
        
        // 2차 필터: BT2
        if (!use_bt2 || checkTreasure(current_seed, bt2_cx, bt2_cz)) {
            int pos = atomicAdd(result_count, 1);
            if (pos < max_results) {
                results[pos] = current_seed;
            }
        }
    }
}

// =========================================================
// [HOST] 메인 함수
// =========================================================

int main() {
    Generator g;
    setupGenerator(&g, MC_VERSION, 0);

    int bt1_cx = BT1_X >> 4;
    int bt1_cz = BT1_Z >> 4;
    int bt2_cx = BT2_X >> 4;
    int bt2_cz = BT2_Z >> 4;
    int sh_cx = SH_X >> 4;
    int sh_cz = SH_Z >> 4;
    bool use_bt2 = (BT2_X != 0 || BT2_Z != 0);

    uint64_t *d_results;
    int *d_count;
    int max_results_buffer = 10000;
    uint64_t *h_results = (uint64_t*)malloc(max_results_buffer * sizeof(uint64_t));
    int h_count = 0;

    cudaMalloc(&d_results, max_results_buffer * sizeof(uint64_t));
    cudaMalloc(&d_count, sizeof(int));

    printf("=== CUDA Seed Cracker for MC 1.%d ===\n", MC_VERSION);
    printf("BT1 Chunk: (%d, %d)\n", bt1_cx, bt1_cz);
    printf("Block/Grid: %d / %d\n", THREADS_PER_BLOCK, BLOCKS_PER_GRID);
    printf("Starting GPU Search...\n");

    uint64_t total_space = 1ULL << 48;
    // Step Size 계산 (오버플로우 방지)
    uint64_t step_size = (uint64_t)BATCH_SIZE * 100; 

    for (uint64_t offset = 0; offset < total_space; offset += step_size) {
        
        cudaMemset(d_count, 0, sizeof(int));

        // 커널 실행
        search_kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(offset, bt1_cx, bt1_cz, bt2_cx, bt2_cz, use_bt2, d_results, d_count, max_results_buffer);
        
        cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

        if (h_count > 0) {
            if (h_count > max_results_buffer) h_count = max_results_buffer;
            cudaMemcpy(h_results, d_results, h_count * sizeof(uint64_t), cudaMemcpyDeviceToHost);

            // [CPU] 3차 필터: Stronghold & Biome
            // OpenMP는 옵션이 없으면 단일 스레드로 돕니다.
            for (int i = 0; i < h_count; i++) {
                uint64_t found_sseed = h_results[i];

                StrongholdIter shIter;
                Pos shPos = initFirstStronghold(&shIter, MC_VERSION, found_sseed);
                int foundStronghold = 0;

                for (int k = 0; k < 12; k++) {
                    if (nextStronghold(&shIter, &g) <= 0) break;
                    int dx = shIter.pos.x - sh_cx;
                    int dz = shIter.pos.z - sh_cz;
                    if (dx * dx + dz * dz < 15 * 15) {
                        foundStronghold = 1;
                        break;
                    }
                }

                if (foundStronghold) {
                    printf("\n[!] Structure Seed Found: %lld\n", (long long)found_sseed);
                    for (uint64_t upper16 = 0; upper16 < 65536; upper16++) {
                        int64_t fullSeed = (int64_t)found_sseed | ((int64_t)upper16 << 48);
                        applySeed(&g, MC_VERSION, fullSeed);
                        int biomeID = getBiomeAt(&g, 1, BT1_X, 64, BT1_Z);
                        printf("   -> Full Seed Candidate: %lld (Biome: %d)\n", fullSeed, biomeID);
                    }
                }
            }
        }

        if ((offset % (total_space / 20)) < step_size) {
            printf("Progress: %.1f%%\r", (double)offset / total_space * 100.0);
        }
    }

    printf("\nSearch finished.\n");
    cudaFree(d_results);
    cudaFree(d_count);
    free(h_results);
    return 0;
}