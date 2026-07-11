#ifndef AVS2DEC_SRC_CPU_H
#define AVS2DEC_SRC_CPU_H

#include <stdint.h>

typedef struct {
    int sse2;
    int ssse3;
    int sse41;
    int avx2;
    int avx512;
    int neon;
} avs2_cpu_flags;

void avs2_cpu_detect(avs2_cpu_flags *flags);
int avs2_cpu_count(void);

#endif
