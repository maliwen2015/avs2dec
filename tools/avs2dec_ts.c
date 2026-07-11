/*
 * avs2dec_ts - 用 FFmpeg 读取 TS 容器, 拆分 AVS2 视频流, 用本解码器解码
 *
 * Usage: avs2dec_ts -i <input.ts> [-o <output.yuv>] [options]
 *
 * 依赖: FFmpeg 8.x (libavformat, libavcodec, libavutil)
 *   C:\msys64\usr\local\mingw64\ffmpeg-8.0.1
 *
 * Build: 由 CMakeLists.txt 中 AVS2DEC_BUILD_CLI_TS 选项控制
 */

#include "avs2dec/avs2dec.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* FFmpeg 头文件 */
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/log.h>

/* 测试用: 禁用 SIMD 的全局开关 (定义在 lf_apply.c) */
extern int g_disable_simd;

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
        "Usage: %s -i <input.ts> [-o <output.yuv>] [options]\n"
        "  -i  input file (TS/MP4/MKV/FLV... any FFmpeg-supported container)\n"
        "  -o  output file (raw YUV or Y4M), '-' for stdout (default: none)\n"
        "  --y4m        write YUV4MPEG2 format\n"
        "  --threads N  number of decode threads (0=auto)\n"
        "  --thread-mode M  0=frame parallel (default), 1=row parallel\n"
        "  --quiet      suppress info messages\n"
        "  --no-filter  skip in-loop filters\n"
        "  --benchmark  decode without writing output, print fps\n"
        "  --frames N   stop after decoding N frames\n"
        "  --no-simd    disable SIMD optimizations\n"
        "  --probe      print stream info and exit (no decode)\n"
        "  -v           verbose (debug)\n"
        "  --version    print version and exit\n",
        prog);
}

/* 写 YUV 帧到文件 */
static int write_yuv_frame(FILE *fp, const avs2_picture *pic)
{
    int bps = pic->bytes_per_sample;
    for (int p = 0; p < 3; p++) {
        int w = pic->width[p];
        int h = pic->height[p];
        const uint8_t *d = pic->data[p];
        ptrdiff_t s = pic->stride[p];
        for (int y = 0; y < h; y++) {
            if (fwrite(d + y * s, 1, (size_t)w * bps, fp) != (size_t)w * bps)
                return -1;
        }
    }
    return 0;
}

/* 写 Y4M 帧头 */
static int write_y4m_header(FILE *fp, const avs2_seq_header *seq)
{
    const char *chroma = (seq->chroma_format == 0) ? "400"
                       : (seq->chroma_format == 1) ? "420"
                       : (seq->chroma_format == 2) ? "422"
                       : "444";
    int bd = (int)seq->internal_bit_depth;
    /* Y4M 的 C 参数: 8-bit 不需要 p 后缀 */
    if (bd > 8) {
        fprintf(fp, "YUV4MPEG2 W%d H%d F%d:1 Ip A0:0 C%sp%d\n",
                seq->enc_width, seq->enc_height,
                (int)seq->frame_rate > 0 ? (int)seq->frame_rate : 25,
                chroma, bd);
    } else {
        fprintf(fp, "YUV4MPEG2 W%d H%d F%d:1 Ip A0:0 C%s\n",
                seq->enc_width, seq->enc_height,
                (int)seq->frame_rate > 0 ? (int)seq->frame_rate : 25,
                chroma);
    }
    return 0;
}

/* 写 Y4M 帧数据 */
static int write_y4m_frame(FILE *fp, const avs2_picture *pic)
{
    fprintf(fp, "FRAME\n");
    return write_yuv_frame(fp, pic);
}

int main(int argc, char *argv[])
{
    const char *in_path = NULL, *out_path = NULL;
    int is_y4m = 0, quiet = 0, verbose = 0, no_filter = 0;
    int benchmark = 0;
    int n_threads = 0;
    int thread_mode = 0;
    int max_frames = 0;
    int no_simd = 0;
    int probe_only = 0;

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
        else if (!strcmp(argv[i], "--probe")) probe_only = 1;
        else if (!strcmp(argv[i], "-v")) verbose = 1;
        else if (!strcmp(argv[i], "--version")) {
            printf("avs2dec_ts (with FFmpeg %s) %s (API 0x%06x)\n",
                   AV_STRINGIFY(LIBAVFORMAT_VERSION),
                   avs2_version(), avs2_version_api());
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

    /* 设置 FFmpeg 日志级别 */
    av_log_set_level(quiet ? AV_LOG_ERROR : (verbose ? AV_LOG_DEBUG : AV_LOG_WARNING));

    /* 1. 用 FFmpeg 打开容器 (TS/MP4/MKV...) */
    AVFormatContext *fmt_ctx = NULL;
    int ret = avformat_open_input(&fmt_ctx, in_path, NULL, NULL);
    if (ret < 0) {
        char err[128];
        av_strerror(ret, err, sizeof(err));
        fprintf(stderr, "无法打开输入文件 %s: %s\n", in_path, err);
        return 1;
    }

    ret = avformat_find_stream_info(fmt_ctx, NULL);
    if (ret < 0) {
        fprintf(stderr, "无法获取流信息\n");
        avformat_close_input(&fmt_ctx);
        return 1;
    }

    /* 2. 查找 AVS2 视频流 (codec_id == AV_CODEC_ID_AVS2) */
    int video_idx = -1;
    const AVCodecParameters *vpar = NULL;
    for (unsigned i = 0; i < fmt_ctx->nb_streams; i++) {
        AVStream *st = fmt_ctx->streams[i];
        if (st->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            /* 优先选 AVS2 流, 否则选第一个视频流 */
            if (st->codecpar->codec_id == AV_CODEC_ID_AVS2) {
                video_idx = i;
                vpar = st->codecpar;
                break;
            }
            if (video_idx < 0) {
                video_idx = i;
                vpar = st->codecpar;
            }
        }
    }

    if (video_idx < 0) {
        fprintf(stderr, "未找到视频流\n");
        avformat_close_input(&fmt_ctx);
        return 1;
    }

    enum AVCodecID cid = fmt_ctx->streams[video_idx]->codecpar->codec_id;
    if (cid != AV_CODEC_ID_AVS2) {
        const char *cname = avcodec_get_name(cid);
        fprintf(stderr, "警告: 视频流编码为 %s (非 AVS2), 将尝试提取裸流解码\n", cname);
    }

    if (!quiet) {
        fprintf(stderr, "输入: %s\n", in_path);
        fprintf(stderr, "容器: %s\n", fmt_ctx->iformat->name);
        fprintf(stderr, "视频流 #%d: %s, %dx%d, %d bit\n",
                video_idx, avcodec_get_name(cid),
                vpar->width, vpar->height,
                vpar->bits_per_coded_sample > 0 ? vpar->bits_per_coded_sample : 8);
        if (vpar->extradata_size > 0) {
            fprintf(stderr, "  extradata: %d bytes\n", vpar->extradata_size);
        }
    }

    if (probe_only) {
        av_dump_format(fmt_ctx, video_idx, in_path, 0);
        avformat_close_input(&fmt_ctx);
        return 0;
    }

    /* 3. 初始化本解码器 */
    avs2_settings s;
    avs2_default_settings(&s);
    s.n_threads = n_threads;
    s.thread_mode = thread_mode;
    s.log_level = quiet ? AVS2_LOG_ERROR : (verbose ? AVS2_LOG_DEBUG : AVS2_LOG_INFO);
    s.skip_loop_filter = no_filter;

    avs2_ctx *ctx = avs2_open(&s);
    if (!ctx) {
        fprintf(stderr, "avs2_open 失败\n");
        avformat_close_input(&fmt_ctx);
        return 1;
    }

    /* 4. 打开输出文件 */
    FILE *out_fp = NULL;
    int y4m_header_written = 0;
    if (!benchmark) {
        if (!out_path) out_path = "-";
        if (strcmp(out_path, "-") == 0) {
            out_fp = stdout;
        } else {
            out_fp = fopen(out_path, "wb");
            if (!out_fp) {
                fprintf(stderr, "无法打开输出文件 %s\n", out_path);
                avs2_close(&ctx);
                avformat_close_input(&fmt_ctx);
                return 1;
            }
        }
    }

    /* 5. 如果有 extradata (AVS2 SPS/序列头), 先送入解码器 */
    int n_frames = 0;
    if (vpar->extradata_size > 0) {
        avs2_data data;
        avs2_data_wrap(&data, vpar->extradata, vpar->extradata_size, 0, 0);
        avs2_send_data(ctx, &data);
        /* drain pictures */
        for (;;) {
            avs2_picture pic; avs2_seq_header seq;
            int r = avs2_get_picture(ctx, &pic, &seq);
            if (r == AVS2_OK) {
                if (out_fp) {
                    if (is_y4m && !y4m_header_written) {
                        write_y4m_header(out_fp, &seq);
                        y4m_header_written = 1;
                    }
                    if (is_y4m) write_y4m_frame(out_fp, &pic);
                    else write_yuv_frame(out_fp, &pic);
                }
                n_frames++;
                avs2_picture_unref(ctx, &pic);
            } else break;
        }
    }

    /* 6. 读取容器中的 AVS2 视频包, 送入解码器 */
    double t_start = get_time_ms();
    AVPacket *pkt = av_packet_alloc();
    int eof = 0;

    while (!eof) {
        if (max_frames > 0 && n_frames >= max_frames) break;

        ret = av_read_frame(fmt_ctx, pkt);
        if (ret < 0) {
            eof = 1;
            break;
        }

        if (pkt->stream_index != video_idx) {
            av_packet_unref(pkt);
            continue;
        }

        /* 将 AVS2 裸流数据送入解码器 */
        avs2_data data;
        avs2_data_wrap(&data, pkt->data, pkt->size,
                       pkt->pts != AV_NOPTS_VALUE ? pkt->pts : n_frames,
                       pkt->dts != AV_NOPTS_VALUE ? pkt->dts : n_frames);
        int r = avs2_send_data(ctx, &data);
        if (r < 0 && !quiet) {
            fprintf(stderr, "send_data 警告: %d\n", r);
        }

        /* 取出已解码的图像 */
        for (;;) {
            if (max_frames > 0 && n_frames >= max_frames) break;
            avs2_picture pic; avs2_seq_header seq;
            r = avs2_get_picture(ctx, &pic, &seq);
            if (r == AVS2_OK) {
                if (out_fp) {
                    if (is_y4m && !y4m_header_written) {
                        write_y4m_header(out_fp, &seq);
                        y4m_header_written = 1;
                    }
                    if (is_y4m) write_y4m_frame(out_fp, &pic);
                    else write_yuv_frame(out_fp, &pic);
                }
                n_frames++;
                avs2_picture_unref(ctx, &pic);
            } else {
                break;
            }
        }

        av_packet_unref(pkt);
    }

    /* 7. flush 解码器 */
    avs2_flush(ctx);
    for (;;) {
        if (max_frames > 0 && n_frames >= max_frames) break;
        avs2_picture pic; avs2_seq_header seq;
        int r = avs2_get_picture(ctx, &pic, &seq);
        if (r == AVS2_OK) {
            if (out_fp) {
                if (is_y4m && !y4m_header_written) {
                    write_y4m_header(out_fp, &seq);
                    y4m_header_written = 1;
                }
                if (is_y4m) write_y4m_frame(out_fp, &pic);
                else write_yuv_frame(out_fp, &pic);
            }
            n_frames++;
            avs2_picture_unref(ctx, &pic);
        } else break;
    }

    double t_end = get_time_ms();
    double elapsed_ms = t_end - t_start;

    /* 8. 清理 */
    av_packet_free(&pkt);
    avformat_close_input(&fmt_ctx);
    if (out_fp && out_fp != stdout) fclose(out_fp);
    avs2_close(&ctx);

    /* 9. 输出统计 */
    if (benchmark) {
        double fps = (elapsed_ms > 0.0) ? (double)n_frames * 1000.0 / elapsed_ms : 0.0;
        fprintf(stderr, "解码 %d 帧, 耗时 %.2f ms (%.2f fps)\n",
                n_frames, elapsed_ms, fps);
    } else if (!quiet) {
        fprintf(stderr, "解码完成, 共 %d 帧.\n", n_frames);
    }

    return 0;
}
