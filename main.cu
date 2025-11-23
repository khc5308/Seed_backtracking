#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// [정석 방법] C 라이브러리는 헤더만 포함하고, extern "C"로 감싸줍니다.
extern "C" {
    #include "finders.h"
    #include "generator.h"
    #include "biomes.h"
    // noise.h나 layers.h는 직접 안 불러도 위 헤더들이 알아서 처리합니다.
}

const int MC_VERSION = MC_1_16_1;

// BT 좌표 1
const int BT1_X = -1831; 
const int BT1_Z = -263;

// BT 좌표 2
const int BT2_X = 0; 
const int BT2_Z = 0;

// 엔더 유적(Stronghold) 좌표
const int SH_X = -1627;
const int SH_Z = 84;

#define THREADS_PER_BLOCK 256
#define BLOCKS_PER_GRID 65536 
#define BATCH_SIZE (THREADS_PER_BLOCK * BLOCKS_PER_GRID)


__device__ __forceinline__ float gpu_nextFloat(uint64_t *seed) {
    *seed = (*seed * 0x5DEECE66DULL + 0xBULL) & ((1ULL << 48) - 1);
    return (int)(*seed >> 24) / ((float)(1 << 24));
}

__device__ bool checkTreasure(uint64_t lower48, int chunkX, int chunkZ) {
    uint64_t seed = lower48 + 10387320ULL;
    seed += (uint64_t)chunkX * 341873128712ULL + (uint64_t)chunkZ * 132897987541ULL;
    seed = (seed ^ 0x5DEECE66DULL) & ((1ULL << 48) - 1);
    return gpu_nextFloat(&seed) < 0.01f;
}

__global__ void search_kernel(uint64_t start_offset, int bt1_cx, int bt1_cz, int bt2_cx, int bt2_cz, 
                              bool use_bt2, uint64_t *results, int *result_count, int max_results) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t current_seed = start_offset + idx;

    if (checkTreasure(current_seed, bt1_cx, bt1_cz)) {
        if (!use_bt2 || checkTreasure(current_seed, bt2_cx, bt2_cz)) {
            int pos = atomicAdd(result_count, 1);
            if (pos < max_results) {
                results[pos] = current_seed;
            }
        }
    }
}

int main() {
    Generator g;
    setupGenerator(&g, MC_VERSION, 0);

    // 좌표 변환
    int bt1_cx = BT1_X >> 4;
    int bt1_cz = BT1_Z >> 4;
    int bt2_cx = BT2_X >> 4;
    int bt2_cz = BT2_Z >> 4;
    

    
    bool use_bt2 = false;

    uint64_t *d_results;
    int *d_count;
    int max_results_buffer = 10000;
    uint64_t *h_results = (uint64_t*)malloc(max_results_buffer * sizeof(uint64_t));
    int h_count = 0;

    cudaMalloc(&d_results, max_results_buffer * sizeof(uint64_t));
    cudaMalloc(&d_count, sizeof(int));

    printf("Starting GPU Search... (Press Ctrl+C to stop)\n");

    uint64_t total_space = 1ULL << 48;
    uint64_t step_size = (uint64_t)BATCH_SIZE; 
    
    int loop_counter = 0;

    for (uint64_t offset = 0; offset < total_space; offset += step_size) {
        
        cudaMemset(d_count, 0, sizeof(int));

        search_kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(offset, bt1_cx, bt1_cz, bt2_cx, bt2_cz, use_bt2, d_results, d_count, max_results_buffer);
        
        cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

        if (h_count > 0) {
            if (h_count > max_results_buffer) h_count = max_results_buffer;
            cudaMemcpy(h_results, d_results, h_count * sizeof(uint64_t), cudaMemcpyDeviceToHost);

            printf("\n[SUCCESS] GPU found %d candidates at offset %lld!\n", h_count, (long long)offset);
            printf("First candidate: %lld\n", (long long)h_results[0]);
            
            printf("System works. Please put correct coordinates and enable full search.\n");
            break; 
        }

        loop_counter++;
        if (loop_counter % 1000 == 0) {
            printf("Scanning... Offset: %lld (No matches yet)\r", (long long)offset);
        }
    }

    printf("\nSearch finished.\n");
    cudaFree(d_results);
    cudaFree(d_count);
    free(h_results);
    return 0;
}