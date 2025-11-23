#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>
#include <math.h> // sqrt를 위해 추가 (디버깅용)
#include "generator.h"
#include "finders.h"

// --- 상수 및 입력 데이터 ---
const int MC_VERSION = MC_1_16_1;

Pos treasures[] = {
    {-407,-151},
    {-631,-151},
    {-1751, -1943}
};
const int TREASURE_COUNT = sizeof(treasures) / sizeof(Pos);

Pos strongholds[] = {
    {-475, -2635}, 
};
const int STRONGHOLD_COUNT = sizeof(strongholds) / sizeof(Pos);

#define NUM_THREADS 12
#define MIN_MATCH_REQUIRED 2 

// LCG 상수
const uint64_t M = 0x5DEECE66DLL;
const uint64_t MASK = (1ULL << 48) - 1;

// --- 구조체 및 전역 변수 ---
typedef struct {
    uint64_t start_k;
    uint64_t end_k;
    int thread_id;
} thread_data_t;

StructureConfig treasureConf;
uint64_t treasure_salts[10]; 


void diagnoseSeed(uint64_t seed) {
    printf("\n[CANDIDATE] Checking Seed: %lld\n", (long long)seed);

    // 1. 보물 검증 (청크 좌표 일치 검사)
    int match_cnt = 0;
    for (int i = 0; i < TREASURE_COUNT; i++) {
        int tx = treasures[i].x >> 4;
        int tz = treasures[i].z >> 4;
        
        // Buried Treasure의 Region Size는 1이므로 Region X/Z는 Chunk X/Z와 같습니다.
        // 다만 getStructurePos는 Region 인자를 요구하므로 인자 전달 규약을 지킵니다.
        int rx = tx; 
        int rz = tz; 

        Pos p = {0, 0}; // 초기화
        // Treasure는 바다 바이옴 체크가 필요 없으므로 getStructurePos만 호출
        int valid = getStructurePos(Treasure, MC_VERSION, seed, rx, rz, &p);
        
        // 생성된 구조물이 존재하고, 그 청크 좌표가 정확히 일치하는지 확인
        if (valid && p.x == tx && p.z == tz) {
            match_cnt++;
        } else {
             printf("  - Treasure #%d (Chunk %d,%d): MISMATCH (Gen Loc: %d,%d)\n", 
                    i, tx, tz, p.x, p.z);
        }
    }
    
    if (match_cnt < MIN_MATCH_REQUIRED) {
        printf("  -> REJECTED (Treasure Matches: %d/%d)\n", match_cnt, TREASURE_COUNT);
        return;
    }
    printf("  -> PASSED Treasure Check (%d/%d)\n", match_cnt, TREASURE_COUNT);

    // 2. 엔더 유적 검증 (좌표 오차 허용 범위 확인)
    Generator g;
    setupGenerator(&g, MC_VERSION, 0);
    applySeed(&g, DIM_OVERWORLD, seed);
    StrongholdIter sh_iter;
    Pos sh_pos = initFirstStronghold(&sh_iter, MC_VERSION, seed);
    
    int stronghold_passed = 1;

    for (int i = 0; i < STRONGHOLD_COUNT; i++) {
        int tx = strongholds[i].x >> 4;
        int tz = strongholds[i].z >> 4;
        int found = 0;
        
        for (int k = 0; k < 128; k++) {
            // 거리 계산 (청크 단위 유클리드 거리 제곱)
            long distSq = (long)(sh_pos.x - tx)*(sh_pos.x - tx) + (long)(sh_pos.z - tz)*(sh_pos.z - tz);
            
            // 오차범위 5청크 내 (25)
            if (distSq < 25) { 
                 printf("  - Stronghold #%d (Target %d,%d): OK (Dist: %.1f chunks, Loc: %d, %d)\n", 
                        i, tx, tz, sqrt((double)distSq), sh_pos.x, sh_pos.z);
                found = 1;
                break;
            }
            if (!nextStronghold(&sh_iter, &g)) break;
            sh_pos = sh_iter.pos;
        }
        
        if (!found) {
            printf("  - Stronghold #%d: NOT FOUND near target %d,%d\n", i, tx, tz);
            stronghold_passed = 0;
            break;
        }
        if (STRONGHOLD_COUNT > 1) sh_pos = initFirstStronghold(&sh_iter, MC_VERSION, seed);
    }
    
    if (stronghold_passed) {
        printf("  -> PASSED Stronghold Check\n");
        // 최종 시드 확정 및 64비트 확장
        for (int upper = 0; upper < (1 << 16); upper++) {
            uint64_t full_seed = seed | ((uint64_t)upper << 48);
            printf("[!!! FINAL 64-BIT SEED !!!] %lld\n", (long long)full_seed);
        }
    }
    printf("--------------------------------------------------\n");
}


// [스레드] 수학적 역산 및 필터링
void *flexible_solver_thread(void *arg) {
    thread_data_t *data = (thread_data_t *)arg;
    uint64_t salt0 = treasure_salts[0];
    
    uint64_t loop_cnt = 0;

    for (uint64_t k = data->start_k; k < data->end_k; k++) {
        
        uint64_t target_high = k * 100; 
        uint64_t base_internal = target_high << 17;

        for (uint64_t low = 0; low < (1 << 17); low++) {
            uint64_t internal_seed = base_internal | low;
            uint64_t world_seed = (internal_seed ^ M) - salt0; 
            world_seed &= MASK;

            // 수학적 필터: 나머지 보물들 중 "몇 개나" 확률을 뚫는지 확인
            int math_matches = 1; // 0번은 이미 맞음
            
            for (int i = 1; i < TREASURE_COUNT; i++) {
                uint64_t check_seed = (world_seed + treasure_salts[i]) & MASK;
                uint64_t internal_check = (check_seed ^ M) & MASK;
                
                if ((internal_check >> 17) % 100 == 0) {
                    math_matches++;
                }
            }
            
            if (math_matches >= MIN_MATCH_REQUIRED) {
                // 수학적 확률을 통과한 시드만 정밀 검사
                diagnoseSeed(world_seed);
            }

            // 진행률 표시
             if (data->thread_id == 0 && (loop_cnt++ & 0x7FFFF) == 0) {
                 printf("."); fflush(stdout);
             }
        }
    }
    pthread_exit(NULL);
}

int main() {
    if (!getStructureConfig(Treasure, MC_VERSION, &treasureConf)) return -1;

    printf("=== Diagnostic Search Mode ===\n");
    printf("Target Chunk Coords:\n");
    
    // Salt 계산 (Salt 계산 시 오버플로우 방지 및 정확한 청크 좌표 사용)
    for(int i=0; i<TREASURE_COUNT; i++) {
        int cx = treasures[i].x >> 4;
        int cz = treasures[i].z >> 4;
        
        // [수정 포인트] int64_t 캐스팅으로 64비트 연산 보장
        int64_t s = (int64_t)cx * 341873128712ULL + (int64_t)cz * 132897987541ULL + (int64_t)treasureConf.salt;
        treasure_salts[i] = s & MASK;
        
        printf(" T#%d Chunk(%d, %d) Salt: %lld\n", i, cx, cz, (long long)treasure_salts[i]);
    }
    printf("Looking for seeds matching at least %d / %d treasures.\n", MIN_MATCH_REQUIRED, TREASURE_COUNT);

    uint64_t MAX_K = (1ULL << 31) / 100;
    pthread_t threads[NUM_THREADS];
    thread_data_t t_data[NUM_THREADS];

    uint64_t chunk = MAX_K / NUM_THREADS;
    for (int i = 0; i < NUM_THREADS; i++) {
        t_data[i].thread_id = i;
        t_data[i].start_k = i * chunk;
        t_data[i].end_k = (i == NUM_THREADS - 1) ? MAX_K : (i + 1) * chunk;
        pthread_create(&threads[i], NULL, flexible_solver_thread, (void *)&t_data[i]);
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    printf("Finished.\n");
    return 0;
}