#include "output.h"
#include <stdint.h>
#include <stdlib.h>

int avs2_output_write_yuv(avs2_output *out, const avs2_picture *pic)
{
    const int bps = pic->bytes_per_sample;

    /* force_8bit: 内部以 10-bit 解码, 输出时降为 8-bit.
     * 逐行转换, 仅需一行缓冲, 无额外帧内存开销. */
    if (out->force_8bit && bps == 2) {
        int max_w = 0;
        uint8_t *row8;
        int pl;

        for (pl = 0; pl < 3; pl++)
            if (pic->width[pl] > max_w) max_w = pic->width[pl];

        row8 = (uint8_t *)malloc(max_w);
        if (!row8) return -1;

        for (pl = 0; pl < 3; pl++) {
            const uint16_t *d16 = (const uint16_t *)(const void *)pic->data[pl];
            ptrdiff_t s = pic->stride[pl];
            int stride16 = (int)(s >> 1);
            int w = pic->width[pl], h = pic->height[pl];
            int y;

            for (y = 0; y < h; y++) {
                const uint16_t *src = d16 + y * stride16;
                int x;
                for (x = 0; x < w; x++) {
                    /* 舍入右移: (v + 2) >> 2, 10-bit → 8-bit */
                    row8[x] = (uint8_t)((src[x] + 2) >> 2);
                }
                if (fwrite(row8, 1, w, out->fp) != (size_t)w) {
                    free(row8);
                    return -1;
                }
            }
        }
        free(row8);
        return 0;
    }

    /* 默认路径: 直接按原始位深输出 */
    for (int pl = 0; pl < 3; pl++) {
        const uint8_t *d = pic->data[pl];
        ptrdiff_t s = pic->stride[pl];
        int w = pic->width[pl], h = pic->height[pl];
        size_t row_bytes = (size_t)w * bps;
        for (int y = 0; y < h; y++)
            if (fwrite(d + y * s, 1, row_bytes, out->fp) != row_bytes)
                return -1;
    }
    return 0;
}
