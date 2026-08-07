#!/bin/bash
# =============================================================================
# avs2dec 多线程性能与位精确验证脚本 (无 pipe 版本)
#
# 用法:
#   ./bench_threads.sh <input.avs2> [选项]
#
# 选项:
#   --threads "1 2 4 8 16"   指定测试的线程数 (默认: 1 2 4 8 16)
#   --modes "0 1"             指定线程模式 0=帧并行 1=行并行 (默认: 0 1)
#   --frames N                只解码前 N 帧 (默认: 0=全部)
#   --no-simd-test            跳过 SIMD 开关对比测试
#   --bitexact-only           只做位精确验证, 跳过性能测试
#   --bench-only              只做性能测试, 跳过位精确验证
#   --binary <path>           指定 avs2dec 路径 (默认: ./build/avs2dec)
#   --output-dir <dir>        结果输出目录 (默认: ./bench_results)
#
# 示例:
#   ./bench_threads.sh 15gop.avs2
#   ./bench_threads.sh 15gop.avs2 --threads "1 2 4" --frames 100
#   ./bench_threads.sh 15gop.avs2 --bitexact-only --threads "1 4"
# =============================================================================

set -eu

# ---- 颜色输出 ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ---- 默认参数 ----
BINARY="./build/avs2dec"
THREADS="1 2 4 8 16"
MODES="0 1"
MAX_FRAMES=0
NO_SIMD_TEST=0
BITEXACT_ONLY=0
BENCH_ONLY=0
OUTPUT_DIR="./bench_results"
INPUT_FILE=""

# ---- 解析命令行 ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --threads)        THREADS="$2"; shift 2 ;;
        --modes)          MODES="$2"; shift 2 ;;
        --frames)         MAX_FRAMES="$2"; shift 2 ;;
        --no-simd-test)   NO_SIMD_TEST=1; shift ;;
        --bitexact-only)  BITEXACT_ONLY=1; shift ;;
        --bench-only)     BENCH_ONLY=1; shift ;;
        --binary)         BINARY="$2"; shift 2 ;;
        --output-dir)     OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help)
            sed -n '3,22p' "$0"
            exit 0 ;;
        *)
            if [[ -z "$INPUT_FILE" ]]; then
                INPUT_FILE="$1"
            else
                echo -e "${RED}错误: 未知参数 '$1'${NC}" >&2
                exit 1
            fi
            shift ;;
    esac
done

# ---- 检查输入 ----
if [[ -z "$INPUT_FILE" ]]; then
    echo -e "${RED}错误: 请指定输入文件${NC}"
    echo "用法: $0 <input.avs2> [选项]"
    exit 1
fi

if [[ ! -f "$INPUT_FILE" ]]; then
    echo -e "${RED}错误: 输入文件不存在: $INPUT_FILE${NC}"
    exit 1
fi

if [[ ! -x "$BINARY" ]]; then
    if [[ -x "./build/avs2dec" ]]; then
        BINARY="./build/avs2dec"
    elif [[ -x "$(dirname "$0")/build/avs2dec" ]]; then
        BINARY="$(dirname "$0")/build/avs2dec"
    else
        echo -e "${RED}错误: 找不到可执行文件: $BINARY${NC}"
        echo "请先编译: cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j\$(nproc)"
        exit 1
    fi
fi

# ---- 创建输出目录和临时目录 ----
mkdir -p "$OUTPUT_DIR"
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$OUTPUT_DIR/bench_${TIMESTAMP}.log"
HASH_REF_FILE="$OUTPUT_DIR/hash_ref_${TIMESTAMP}.txt"
DECODE_OUT="$TMPDIR/decode_out.txt"
DIFF_FILE="$TMPDIR/diff_out.txt"

# ---- 帧数参数 ----
FRAMES_ARG=""
if [[ "$MAX_FRAMES" -gt 0 ]]; then
    FRAMES_ARG="--frames $MAX_FRAMES"
fi

# ---- 辅助函数 ----

# 模式名
mode_name() {
    if [[ "$1" == "0" ]]; then
        echo "帧并行"
    else
        echo "行并行"
    fi
}

# 日志输出 (写入文件 + 终端, 不使用 pipe)
log() {
    echo -e "$@" >> "$LOG_FILE"
    echo -e "$@"
}

# 格式化日志输出 (printf 版本, 不使用 pipe)
log_printf() {
    printf "$@" >> "$LOG_FILE"
    printf "$@"
}

# 浮点除法 (使用 awk, 不使用 pipe)
calc_ratio() {
    awk "BEGIN {printf \"%.2f\", $1 / $2}"
}

# 运行单次解码测试
run_decode() {
    local desc="$1"
    local threads="$2"
    local mode="$3"
    local extra_args="$4"
    local result_file="$5"

    local mname
    mname=$(mode_name "$mode")

    log "${CYAN}▶ ${BOLD}$desc${NC} ${CYAN}| 线程=$threads 模式=$mname($mode)${NC}"

    local start_ts end_ts elapsed
    start_ts=$(date +%s.%N)

    # 运行解码器, 输出重定向到临时文件 (不使用 pipe)
    local rc=0
    "$BINARY" -i "$INPUT_FILE" --benchmark \
        --threads "$threads" --thread-mode "$mode" \
        $FRAMES_ARG $extra_args > "$DECODE_OUT" 2>&1 || rc=$?

    end_ts=$(date +%s.%N)
    elapsed=$(awk "BEGIN {printf \"%.2f\", $end_ts - $start_ts}")

    if [[ "$rc" -ne 0 ]]; then
        log "  ${RED}✗ 解码失败 (返回码=$rc)${NC}"
        local errline
        errline=$(head -5 "$DECODE_OUT")
        log "  输出: $errline"
        echo "FAIL,$desc,$threads,$mode,$extra_args,0.00,0" >> "$result_file"
        return 1
    fi

    # 从输出中提取 fps 和帧数 (使用 bash 正则, 不使用 pipe)
    local output fps frames
    output=$(cat "$DECODE_OUT")
    fps="0.00"
    frames="0"

    if [[ "$output" =~ ([0-9]+\.[0-9]+)\ fps ]]; then
        fps="${BASH_REMATCH[1]}"
    fi
    if [[ "$output" =~ Decoded\ ([0-9]+)\ frames ]]; then
        frames="${BASH_REMATCH[1]}"
    fi

    log "  ${GREEN}✓ ${BOLD}${fps} fps${NC}  ${frames} 帧  ${elapsed}s"
    echo "OK,$desc,$threads,$mode,$extra_args,$fps,$frames" >> "$result_file"
    return 0
}

# =============================================================================
# 打印表头
# =============================================================================
log "================================================================"
log "  avs2dec 多线程性能测试"
log "  时间: $(date '+%Y-%m-%d %H:%M:%S')"
log "  二进制: $BINARY"

# 文件大小 (不使用 pipe: du + bash 参数展开)
file_size_raw=$(du -h "$INPUT_FILE")
file_size=${file_size_raw%%$'\t'*}
log "  输入: $INPUT_FILE ($file_size)"
log "  线程数: $THREADS"
log "  模式: $MODES (0=帧并行, 1=行并行)"
if [[ "$MAX_FRAMES" -gt 0 ]]; then
    log "  帧数限制: $MAX_FRAMES"
fi
log "  日志: $LOG_FILE"
log "================================================================"
echo "" >> "$LOG_FILE"
echo ""

# ---- 环境信息 ----
log "${BOLD}[环境信息]${NC}"

# 版本信息 (不使用 pipe: 重定向到临时文件)
"$BINARY" --version > "$DECODE_OUT" 2>&1
cat "$DECODE_OUT" >> "$LOG_FILE"
cat "$DECODE_OUT"
echo "" >> "$LOG_FILE"
echo ""

# CPU 信息 (使用 awk 读取 /proc/cpuinfo, 不使用 pipe)
if [[ -f /proc/cpuinfo ]]; then
    cpu_model=$(awk -F': ' '/^model name/{print $2; exit}' /proc/cpuinfo)
    cpu_cores=$(nproc 2>/dev/null || echo "?")
    log "CPU: ${cpu_model:-未知}"
    log "核心数: $cpu_cores"

    if [[ $(uname -m) == "aarch64" ]]; then
        if grep -q 'asimddp' /proc/cpuinfo 2>/dev/null; then
            log "SDOT (dotprod): ${GREEN}支持${NC}"
        else
            log "SDOT (dotprod): ${YELLOW}不支持 (将回退蝶形 NEON)${NC}"
        fi
    fi
fi
echo "" >> "$LOG_FILE"
echo ""

# =============================================================================
# 第一部分: 性能测试
# =============================================================================
if [[ "$BITEXACT_ONLY" -eq 0 ]]; then
    log "${BOLD}${BLUE}========== 第一部分: 多线程性能测试 ==========${NC}"
    echo "" >> "$LOG_FILE"
    echo ""

    PERF_FILE="$OUTPUT_DIR/perf_${TIMESTAMP}.csv"
    echo "status,desc,threads,mode,extra,fps,frames" > "$PERF_FILE"

    for mode in $MODES; do
        mname=$(mode_name "$mode")
        log "${BOLD}--- $mname ---${NC}"

        for t in $THREADS; do
            run_decode "多线程" "$t" "$mode" "" "$PERF_FILE" || true
        done
        echo "" >> "$LOG_FILE"
        echo ""
    done

    # ---- SIMD 对比测试 ----
    if [[ "$NO_SIMD_TEST" -eq 0 ]]; then
        log "${BOLD}--- SIMD 开关对比 (单线程) ---${NC}"
        run_decode "SIMD开启" "1" "0" "" "$PERF_FILE" || true
        run_decode "SIMD关闭" "1" "0" "--no-simd" "$PERF_FILE" || true
        echo "" >> "$LOG_FILE"
        echo ""
    fi

    # ---- 10-bit vs 8-bit 对比 ----
    log "${BOLD}--- 10-bit vs 8-bit 对比 (单线程) ---${NC}"
    run_decode "10-bit原生" "1" "0" "" "$PERF_FILE" || true
    run_decode "强制8-bit" "1" "0" "--8bit" "$PERF_FILE" || true
    echo "" >> "$LOG_FILE"
    echo ""

    # ---- 性能汇总表 ----
    log "${BOLD}${BLUE}========== 性能汇总 ==========${NC}"
    echo "" >> "$LOG_FILE"
    echo ""

    log_printf "%-8s %-8s %-10s %-10s\n" "线程数" "模式" "FPS" "加速比"
    log_printf "%-8s %-8s %-10s %-10s\n" "------" "----" "---" "------"

    # 获取单线程基准 fps (模式 0) — 使用 awk 读取 CSV, 不使用 pipe
    base_fps=$(awk -F',' '/^OK,多线程,1,0,/{print $6; exit}' "$PERF_FILE")
    if [[ -z "$base_fps" ]] || [[ "$base_fps" == "0.00" ]]; then
        base_fps="1.00"
        log "${YELLOW}警告: 无法获取单线程基准 FPS, 加速比将不可靠${NC}"
    fi

    # 遍历性能数据 (使用 while read < file, 不使用 pipe)
    while IFS=',' read -r status desc threads mode extra fps frames; do
        [[ "$status" == "status" ]] && continue
        [[ "$status" == "FAIL" ]] && continue
        [[ "$desc" != "多线程" ]] && continue

        mlabel=$(mode_name "$mode")
        speedup=$(calc_ratio "$fps" "$base_fps")

        log_printf "%-8s %-8s %-10s %-10s\n" "$threads" "$mlabel" "$fps" "x${speedup}"
    done < "$PERF_FILE"

    echo "" >> "$LOG_FILE"
    echo ""

    # SIMD 对比汇总 (使用 awk 提取, 不使用 pipe)
    if [[ "$NO_SIMD_TEST" -eq 0 ]]; then
        simd_on=$(awk -F',' '/^OK,SIMD开启,/{print $6; exit}' "$PERF_FILE")
        simd_off=$(awk -F',' '/^OK,SIMD关闭,/{print $6; exit}' "$PERF_FILE")
        if [[ -n "$simd_on" ]] && [[ -n "$simd_off" ]] && [[ "$simd_off" != "0.00" ]]; then
            simd_speedup=$(calc_ratio "$simd_on" "$simd_off")
            log "SIMD 加速比: ${BOLD}x${simd_speedup}${NC} (开启=${simd_on} fps, 关闭=${simd_off} fps)"
        fi
    fi

    # 10-bit vs 8-bit 汇总
    bit10_fps=$(awk -F',' '/^OK,10-bit原生,/{print $6; exit}' "$PERF_FILE")
    bit8_fps=$(awk -F',' '/^OK,强制8-bit,/{print $6; exit}' "$PERF_FILE")
    if [[ -n "$bit10_fps" ]] && [[ -n "$bit8_fps" ]] && [[ "$bit10_fps" != "0.00" ]]; then
        bit_ratio=$(calc_ratio "$bit8_fps" "$bit10_fps")
        log "8-bit/10-bit 速度比: ${BOLD}x${bit_ratio}${NC} (10-bit=${bit10_fps} fps, 8-bit=${bit8_fps} fps)"
    fi

    echo "" >> "$LOG_FILE"
    echo ""
    log "详细数据: $PERF_FILE"
    echo "" >> "$LOG_FILE"
    echo ""
fi

# =============================================================================
# 第二部分: 位精确验证
# =============================================================================
if [[ "$BENCH_ONLY" -eq 0 ]]; then
    log "${BOLD}${BLUE}========== 第二部分: 位精确验证 ==========${NC}"
    echo "" >> "$LOG_FILE"
    echo ""

    log "验证方法: 对比单线程与多线程的逐帧 FNV-1a 哈希值"
    echo "" >> "$LOG_FILE"
    echo ""

    # 生成单线程参考哈希
    log "${CYAN}▶ 生成单线程参考哈希 (threads=1)...${NC}"
    if "$BINARY" -i "$INPUT_FILE" --benchmark --frame-hash \
       --threads 1 --thread-mode 0 $FRAMES_ARG > "$HASH_REF_FILE" 2>/dev/null; then
        ref_count=$(awk '/^FRAME /{n++} END{print n+0}' "$HASH_REF_FILE")
        log "  ${GREEN}✓ 参考哈希生成完成: ${ref_count} 帧${NC}"
    else
        log "  ${RED}✗ 参考哈希生成失败${NC}"
        exit 1
    fi
    echo "" >> "$LOG_FILE"
    echo ""

    # 对每个线程数和模式进行验证
    BITEXACT_FILE="$OUTPUT_DIR/bitexact_${TIMESTAMP}.csv"
    echo "threads,mode,status,mismatch_frame,mismatch_count" > "$BITEXACT_FILE"

    for mode in $MODES; do
        for t in $THREADS; do
            # 跳过 1线程帧并行 (即参考本身)
            [[ "$t" == "1" ]] && [[ "$mode" == "0" ]] && continue

            mlabel=$(mode_name "$mode")
            log "${CYAN}▶ 验证 threads=$t $mlabel...${NC}"

            local_hash="$OUTPUT_DIR/hash_t${t}_m${mode}_${TIMESTAMP}.txt"
            if ! "$BINARY" -i "$INPUT_FILE" --benchmark --frame-hash \
                --threads "$t" --thread-mode "$mode" $FRAMES_ARG > "$local_hash" 2>/dev/null; then
                log "  ${RED}✗ 解码失败${NC}"
                echo "$t,$mode,FAIL,-1,-1" >> "$BITEXACT_FILE"
                continue
            fi

            # 对比哈希: diff 返回 0=相同, 1=不同, 2=错误
            # 使用 if 处理, 避免 set -e 在 diff 返回 1 时退出
            if diff "$HASH_REF_FILE" "$local_hash" > "$DIFF_FILE" 2>/dev/null; then
                # diff 返回 0 = 完全一致
                log "  ${GREEN}✓ 位精确一致 (所有帧匹配)${NC}"
                echo "$t,$mode,PASS,0,0" >> "$BITEXACT_FILE"
            else
                # diff 返回 1 = 有差异
                # 提取首个不匹配帧号 (使用 awk 读取 diff 文件)
                mismatch_frame=$(awk '/^< FRAME/{print $2; exit}' "$DIFF_FILE")
                [[ -z "$mismatch_frame" ]] && mismatch_frame="?"

                # 统计不匹配数 (使用 awk 计数, 避免 grep -c 返回 1 触发 set -e)
                mismatch_count=$(awk '/^< FRAME/{n++} END{print n+0}' "$DIFF_FILE")

                log "  ${RED}✗ 位精确不一致: 首个不匹配帧=$mismatch_frame, 共 $mismatch_count 帧不匹配${NC}"

                # 显示前 3 个差异 (使用 while read < file, 不使用 pipe)
                local n=0
                while IFS= read -r line; do
                    [[ "$line" != "< "* ]] && continue
                    n=$((n + 1))
                    [[ $n -gt 3 ]] && break
                    log "    参考: ${line#< }"
                done < "$DIFF_FILE"

                n=0
                while IFS= read -r line; do
                    [[ "$line" != "> "* ]] && continue
                    n=$((n + 1))
                    [[ $n -gt 3 ]] && break
                    log "    实际: ${line#> }"
                done < "$DIFF_FILE"

                echo "$t,$mode,FAIL,$mismatch_frame,$mismatch_count" >> "$BITEXACT_FILE"
            fi
        done
    done

    echo "" >> "$LOG_FILE"
    echo ""

    # 位精确汇总表
    log "${BOLD}${BLUE}========== 位精确验证汇总 ==========${NC}"
    echo "" >> "$LOG_FILE"
    echo ""

    log_printf "%-8s %-8s %-8s %-15s %-10s\n" "线程数" "模式" "结果" "首个不匹配帧" "不匹配数"
    log_printf "%-8s %-8s %-8s %-15s %-10s\n" "------" "----" "----" "------------" "--------"

    while IFS=',' read -r threads mode status mismatch_frame mismatch_count; do
        [[ "$threads" == "threads" ]] && continue

        mlabel=$(mode_name "$mode")

        if [[ "$status" == "PASS" ]]; then
            log_printf "%-8s %-8s %-8s %-15s %-10s\n" \
                "$threads" "$mlabel" "PASS" "-" "-"
        else
            log_printf "%-8s %-8s %-8s %-15s %-10s\n" \
                "$threads" "$mlabel" "FAIL" "$mismatch_frame" "$mismatch_count"
        fi
    done < "$BITEXACT_FILE"

    echo "" >> "$LOG_FILE"
    echo ""
    log "详细数据: $BITEXACT_FILE"
    log "参考哈希: $HASH_REF_FILE"
    echo "" >> "$LOG_FILE"
    echo ""

    # 清理临时哈希文件 (保留参考哈希)
    for mode in $MODES; do
        for t in $THREADS; do
            [[ "$t" == "1" ]] && [[ "$mode" == "0" ]] && continue
            rm -f "$OUTPUT_DIR/hash_t${t}_m${mode}_${TIMESTAMP}.txt"
        done
    done
fi

# =============================================================================
# 最终汇总
# =============================================================================
log "${BOLD}${BLUE}==============================================${NC}"
log "${BOLD}  测试完成!${NC}"
log "${BOLD}==============================================${NC}"
log "完整日志: $LOG_FILE"
log "结果目录: $OUTPUT_DIR/"
echo ""

# 打印关键结论
if [[ "$BITEXACT_ONLY" -eq 0 ]] && [[ -f "${PERF_FILE:-}" ]]; then
    log "${BOLD}关键指标:${NC}"
    log "  单线程基准 (帧并行): ${base_fps} fps"

    # 使用 awk 找最高 fps 的行 (不使用 pipe / sort / head)
    best_line=$(awk -F',' '
        /^OK,多线程,/ {
            if ($6+0 > max+0) { max = $6+0; line = $0 }
        }
        END { print line }
    ' "$PERF_FILE")

    if [[ -n "$best_line" ]]; then
        # 写入临时文件后用 awk 提取字段 (不使用 pipe)
        echo "$best_line" > "$TMPDIR/best.txt"
        best_threads=$(awk -F',' '{print $3}' "$TMPDIR/best.txt")
        best_mode=$(awk -F',' '{print $4}' "$TMPDIR/best.txt")
        best_fps=$(awk -F',' '{print $6}' "$TMPDIR/best.txt")
        best_mlabel=$(mode_name "$best_mode")
        if [[ -n "$best_fps" ]] && [[ "$best_fps" != "0.00" ]]; then
            log "  最佳配置: threads=$best_threads $best_mlabel = ${BOLD}${best_fps} fps${NC}"
        fi
    fi
fi
