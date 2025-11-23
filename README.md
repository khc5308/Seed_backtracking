gcc -O3 -o seed_cracker seed_cracker_core.c -lpthread

930280484323162760


// 버전 설정 (MC_1_20 등)
const int MC_VERSION = MC_1_16_1;

// 매몰된 보물(Buried Treasure) 좌표 1 (필수)
// F3 -> Targeted Block 좌표 기준
const int BT1_X = 1234; 
const int BT1_Z = -5678;

// 매몰된 보물 좌표 2 (선택: 사용 안 하면 0)
// 0이 아니면 속도가 훨씬 빨라짐
const int BT2_X = 4321; 
const int BT2_Z = -8765;

// 엔더 유적(Stronghold) 좌표
const int SH_X = -1500;
const int SH_Z = 300;