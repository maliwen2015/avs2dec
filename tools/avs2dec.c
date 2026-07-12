/*
 * avs2dec - command line AVS2 decoder tool
 *
 * Usage: avs2dec [-i input] [-o output] [--y4m] [--threads N] [--quiet]
 *
 * Reads AVS2 Annex B bitstream, decodes, and writes raw YUV or Y4M.
 */

#include "avs2dec/avs2dec.h"
#include "input/input.h"
#include "output/output.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* 测试用: 禁用 SIMD 的全局开关 (定义在 lf_apply.c) */
extern int g_disable_simd;

/* 逐帧 FNV-1a 64位哈希 (用于定位 pipeline 首个出错帧) */
static uint64_t fnv1a_frame_hash(const avs2_picture *pic)
{
    uint64_t hash = 14695981039346656037ULL;
    for (int p = 0; p < 3; p++) {
        const uint8_t *d = pic->data[p];
        ptrdiff_t stride = pic->stride[p];
        int w = pic->width[p];
        int h = pic->height[p];
        int bps = pic->bytes_per_sample;
        for (int y = 0; y < h; y++) {
            const uint8_t *row = d + y * stride;
            for (int i = 0; i < w * bps; i++) {
                hash ^= row[i];
                hash *= 1099511628211ULL;
            }
        }
    }
    return hash;
}

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#else
#include <time.h>
#endif

/* 高精度计时 (返回毫秒) */
static double get_time_ms(void)
{
#if defined(_WIN32) || defined(_WIN64)
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart / (double)freq.QuadPart * 1000.0;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
#endif
}

static void usage(const char *prog)
{
    fprintf(stderr,
        "Usage: %s -i <input.avs2> [-o <output.yuv>] [options]\n"
        "  -i  input file (Annex B), '-' for stdin\n"
        "  -o  output file, '-' for stdout (default: none)\n"
        "  --y4m        write YUV4MPEG2 format\n"
        "  --threads N  number of decode threads (0=auto)\n"
        "  --thread-mode M  0=frame parallel (default), 1=row parallel\n"
        "  --quiet      suppress info messages\n"
        "  --no-filter  skip in-loop filters\n"
        "  --benchmark  decode without writing output, print fps\n"
        "  --frames N   stop after decoding N frames\n"
        "  --no-simd    disable SIMD optimizations\n"
        "  --8bit       force 8-bit decode (lossy, faster)\n"
        "  --frame-hash  print per-frame FNV-1a hash (for diff debugging)\n"
        "  -v           verbose (debug)\n"
        "  --version    print version and exit\n",
        prog);
}

int main(int argc, char *argv[])
{
    const char *in_path = NULL, *out_path = NULL;
    int is_y4m = 0, quiet = 0, verbose = 0, no_filter = 0;
    int benchmark = 0;
    int n_threads = 0;
    int thread_mode = 0;  /* 0=frame, 1=row */
    int max_frames = 0;  /* 0 = unlimited */
    int no_simd = 0;
    int frame_hash = 0;  /* 逐帧哈希输出 */
    int force_8bit = 0;  /* 强制 8-bit 解码 */

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-i") && i + 1 < argc) in_path = argv[++i];
        else if (!strcmp(argv[i], "-o") && i + 1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "--y4m")) is_y4m = 1;
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) n_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--thread-mode") && i + 1 < argc) thread_mode = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--quiet")) quiet = 1;
        else if (!strcmp(argv[i], "--no-filter")) no_filter = 1;
        else if (!strcmp(argv[i], "--benchmark")) benchmark = 1;
        else if (!strcmp(argv[i], "--frames") && i + 1 < argc) max_frames = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-simd")) no_simd = 1;
        else if (!strcmp(argv[i], "--8bit")) force_8bit = 1;
        else if (!strcmp(argv[i], "--frame-hash")) { frame_hash = 1; benchmark = 1; }
        else if (!strcmp(argv[i], "-v")) verbose = 1;
        else if (!strcmp(argv[i], "--version")) {
            printf("avs2dec %s (API 0x%06x)\n", avs2_version(), avs2_version_api());
            return 0;
        } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            usage(argv[0]); return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]); return 1;
        }
    }
    if (!in_path) { usage(argv[0]); return 1; }

    /* 设置全局开关 (必须在 avs2_open 之前) */
    g_disable_simd = no_simd;

    avs2_settings s;
    avs2_default_settings(&s);
    s.n_threads = n_threads;
    s.thread_mode = thread_mode;
    s.log_level = quiet ? AVS2_LOG_ERROR : (verbose ? AVS2_LOG_DEBUG : AVS2_LOG_INFO);
    s.skip_loop_filter = no_filter;
    s.force_8bit = force_8bit;

    avs2_ctx *ctx = avs2_open(&s);
    if (!ctx) { fprintf(stderr, "avs2_open failed\n"); return 1; }

    avs2_input in;
    if (avs2_input_open(&in, in_path, 1) < 0) {
        fprintf(stderr, "cannot open input %s\n", in_path);
        avs2_close(&ctx);
        return 1;
    }

    /* benchmark 模式不需要输出文件 */
    avs2_output out;
    int has_output = 0;
    if (!benchmark) {
        if (!out_path) out_path = "-";  /* 默认 stdout */
        if (avs2_output_open(&out, out_path, is_y4m) < 0) {
            fprintf(stderr, "cannot open output\n");
            avs2_input_close(&in);
            avs2_close(&ctx);
            return 1;
        }
        out.force_8bit = force_8bit;
        has_output = 1;
    }

    int n_frames = 0;
    uint8_t buf[1 << 16];
    int eof = 0;

    double t_start = get_time_ms();

    while (!eof) {
        if (max_frames > 0 && n_frames >= max_frames) break;
        int rd = avs2_input_read(&in, buf, (int)sizeof(buf));
        if (rd <= 0) { eof = 1; break; }
        avs2_data data;
        avs2_data_wrap(&data, buf, rd, n_frames, n_frames);
        int r = avs2_send_data(ctx, &data);
        if (r < 0) fprintf(stderr, "send_data: %d\n", r);

        /* drain pictures */
        for (;;) {
            if (max_frames > 0 && n_frames >= max_frames) break;
            avs2_picture pic; avs2_seq_header seq;
            r = avs2_get_picture(ctx, &pic, &seq);
            if (r == AVS2_OK) {
                if (frame_hash) {
                    uint64_t h = fnv1a_frame_hash(&pic);
                    fprintf(stderr, "FRAME %d POC %d HASH %016llx\n", n_frames, pic.poc, (unsigned long long)h);
                }
                if (has_output) {
                    if (is_y4m) {
                        extern int avs2_output_write_y4m(avs2_output *, const avs2_picture *, const avs2_seq_header *);
                        avs2_output_write_y4m(&out, &pic, &seq);
                    } else {
                        extern int avs2_output_write_yuv(avs2_output *, const avs2_picture *);
                        avs2_output_write_yuv(&out, &pic);
                    }
                }
                n_frames++;
                avs2_picture_unref(ctx, &pic);
            } else {
                break;
            }
        }
    }

    /* flush: 交替输出帧和解码剩余缓冲区.
     * DPB 满时 avs2_send_data 会返回, 需要先 avs2_get_picture 输出帧释放 DPB 空间,
     * 再调 avs2_send_data(NULL) 重试解码. */
    avs2_flush(ctx);
    for (;;) {
        if (max_frames > 0 && n_frames >= max_frames) break;
        /* 先输出所有可用帧 */
        for (;;) {
            if (max_frames > 0 && n_frames >= max_frames) break;
            avs2_picture pic; avs2_seq_header seq;
            int r = avs2_get_picture(ctx, &pic, &seq);
            if (r == AVS2_OK) {
                if (frame_hash) {
                    uint64_t h = fnv1a_frame_hash(&pic);
                    fprintf(stderr, "FRAME %d POC %d HASH %016llx\n", n_frames, pic.poc, (unsigned long long)h);
                }
                if (has_output) {
                    if (is_y4m) {
                        extern int avs2_output_write_y4m(avs2_output *, const avs2_picture *, const avs2_seq_header *);
                        avs2_output_write_y4m(&out, &pic, &seq);
                    } else {
                        extern int avs2_output_write_yuv(avs2_output *, const avs2_picture *);
                        avs2_output_write_yuv(&out, &pic);
                    }
                }
                n_frames++;
                avs2_picture_unref(ctx, &pic);
            } else break;
        }
        /* 尝试解码更多帧 (从剩余缓冲区) */
        int r = avs2_send_data(ctx, NULL);
        if (r != AVS2_OK) break;
        /* 检查是否还有未解码的数据: 若 avs2_get_picture 返回 EOF 且
         * avs2_send_data 没有新进展, 退出. */
        avs2_picture pic; avs2_seq_header seq;
        int ck = avs2_get_picture(ctx, &pic, &seq);
        if (ck == AVS2_OK) {
            if (frame_hash) {
                uint64_t h = fnv1a_frame_hash(&pic);
                fprintf(stderr, "FRAME %d POC %d HASH %016llx\n", n_frames, pic.poc, (unsigned long long)h);
            }
            if (has_output) {
                if (is_y4m) {
                    extern int avs2_output_write_y4m(avs2_output *, const avs2_picture *, const avs2_seq_header *);
                    avs2_output_write_y4m(&out, &pic, &seq);
                } else {
                    extern int avs2_output_write_yuv(avs2_output *, const avs2_picture *);
                    avs2_output_write_yuv(&out, &pic);
                }
            }
            n_frames++;
            avs2_picture_unref(ctx, &pic);
        } else {
            break;  /* 没有更多帧可输出或解码 */
        }
    }

    double t_end = get_time_ms();
    double elapsed_ms = t_end - t_start;

    avs2_input_close(&in);
    if (has_output) avs2_output_close(&out);

    if (benchmark) {
        double fps = (elapsed_ms > 0.0) ? (double)n_frames * 1000.0 / elapsed_ms : 0.0;
        fprintf(stderr, "Decoded %d frames in %.2f ms (%.2f fps)\n",
                n_frames, elapsed_ms, fps);
    } else if (!quiet) {
        fprintf(stderr, "Decoded %d frames.\n", n_frames);
    }
    avs2_close(&ctx);
    return 0;
}
