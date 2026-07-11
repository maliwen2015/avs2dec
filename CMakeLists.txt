cmake_minimum_required(VERSION 3.13)
project(avs2dec C)

set(AVS2DEC_VERSION_MAJOR 1)
set(AVS2DEC_VERSION_MINOR 0)
set(AVS2DEC_VERSION_PATCH 0)
set(AVS2DEC_VERSION "${AVS2DEC_VERSION_MAJOR}.${AVS2DEC_VERSION_MINOR}.${AVS2DEC_VERSION_PATCH}")

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(AVS2DEC_BUILD_CLI "Build the avs2dec command line tool" ON)
option(AVS2DEC_BUILD_SHARED "Build shared library" OFF)

if(MSVC)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
    add_compile_options(/W3 /utf-8)
else()
    add_compile_options(-Wall -Wextra -Wno-unused-parameter -Wno-unused-function)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3 -march=native -flto -funroll-loops)
    endif()
endif()

if(NOT MSVC AND CMAKE_BUILD_TYPE STREQUAL "Release")
    add_link_options(-flto)
endif()

find_package(Threads REQUIRED)

set(AVS2DEC_SOURCES
    src/log.c
    src/mem.c
    src/cpu.c
    src/getbits.c
    src/aec.c
    src/tables.c
    src/stream.c
    src/header.c
    src/picture.c
    src/ref.c
    src/lib.c
    src/decode.c
    src/cu.c
    src/ipred.c
    src/mc.c
    src/itx.c
    src/quant.c
    src/loopfilter.c
    src/sao.c
    src/alf.c
    src/lf_apply.c
    src/simd_init.c
    src/itx_simd.c
    src/mc_simd.c
    src/ipred_simd.c
    src/lf_simd.c
    src/sao_simd.c
    src/alf_simd.c
    src/quant_simd.c
)

set(AVS2DEC_HEADERS
    include/avs2dec/avs2dec.h
    include/avs2dec/common.h
    include/avs2dec/picture.h
    include/avs2dec/data.h
    include/avs2dec/headers.h
    include/avs2dec/version.h
)

add_library(avs2dec ${AVS2DEC_SOURCES})
target_include_directories(avs2dec
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(avs2dec PRIVATE Threads::Threads)

# SIMD 源文件包含 AVX2/SSE4.1 intrinsics, 需要对应的编译选项
# simd_init.c 不含 intrinsics, 不需要特殊标志
set(AVS2DEC_SIMD_SOURCES
    src/itx_simd.c
    src/mc_simd.c
    src/ipred_simd.c
    src/lf_simd.c
    src/sao_simd.c
    src/alf_simd.c
    src/quant_simd.c
)
if(MSVC)
    set_source_files_properties(${AVS2DEC_SIMD_SOURCES} PROPERTIES
        COMPILE_OPTIONS "/arch:AVX2"
    )
else()
    # GCC/MinGW: 启用 AVX2 + FMA + SSE4.1
    # AVX2 代码必须用 -mavx2 编译, 否则 intrinsics 不可用
    set_source_files_properties(${AVS2DEC_SIMD_SOURCES} PROPERTIES
        COMPILE_OPTIONS "-mavx2;-mfma;-msse4.1"
    )
endif()

if(AVS2DEC_BUILD_SHARED)
    target_compile_definitions(avs2dec PRIVATE AVS2DEC_BUILD_EXPORTS)
    set_target_properties(avs2dec PROPERTIES
        VERSION ${AVS2DEC_VERSION}
        SOVERSION ${AVS2DEC_VERSION_MAJOR}
        C_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
    )
else()
    target_compile_definitions(avs2dec PUBLIC AVS2DEC_STATIC)
endif()

if(AVS2DEC_BUILD_CLI)
    add_executable(avs2dec_cli
        tools/avs2dec.c
        tools/input/input.c
        tools/input/annexb.c
        tools/input/section5.c
        tools/output/output.c
        tools/output/yuv.c
        tools/output/y4m.c
    )
    target_include_directories(avs2dec_cli PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/tools
    )
    target_link_libraries(avs2dec_cli PRIVATE avs2dec Threads::Threads)
    set_target_properties(avs2dec_cli PROPERTIES OUTPUT_NAME avs2dec)
endif()

if(AVS2DEC_BUILD_CLI)
    add_executable(gop_split
        tools/gop_split.c
    )
    target_include_directories(gop_split PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/tools
    )
    set_target_properties(gop_split PROPERTIES OUTPUT_NAME gop_split)
endif()

# ============================================================================
# avs2dec_ts: FFmpeg 容器解封装 + AVS2 解码
# ============================================================================
option(AVS2DEC_BUILD_CLI_TS "Build the FFmpeg-based TS/MKV/MP4 decoder tool" OFF)

set(FFMPEG_ROOT "" CACHE PATH "FFmpeg installation root (containing include/ and lib/)")

if(AVS2DEC_BUILD_CLI_TS)
    # 查找 FFmpeg 库
    if(NOT FFMPEG_ROOT)
        set(FFMPEG_ROOT "C:/msys64/usr/local/mingw64/ffmpeg-8.0.1")
    endif()

    set(FFMPEG_INCLUDE_DIR "${FFMPEG_ROOT}/include")
    set(FFMPEG_LIB_DIR "${FFMPEG_ROOT}/lib")

    add_executable(avs2dec_ts_cli
        tools/avs2dec_ts.c
    )
    target_include_directories(avs2dec_ts_cli PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/tools
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${FFMPEG_INCLUDE_DIR}
    )
    target_link_directories(avs2dec_ts_cli PRIVATE 
    ${FFMPEG_LIB_DIR}
    "C:/msys64/usr/local/mingw64/lib")
    target_link_options(avs2dec_ts_cli PRIVATE
        "-Wl,--start-group"
    )
    target_link_libraries(avs2dec_ts_cli PRIVATE
        avs2dec
        Threads::Threads
        ${FFMPEG_LIB_DIR}/libavformat.a
        ${FFMPEG_LIB_DIR}/libavcodec.a
        ${FFMPEG_LIB_DIR}/libavutil.a
        ${FFMPEG_LIB_DIR}/libswresample.a
        ${FFMPEG_LIB_DIR}/libswscale.a
        ws2_32 secur32 bcrypt
        z bz2 lzma zstd
        ssl crypto
        ogg vorbis vorbisenc
        opus
        mp3lame
        x264 x265 stdc++
        fdk-aac
        vpx
        ass fontconfig freetype
        xml2
        iconv
        intl
        ffi
        harfbuzz glib-2.0
        uuid
        va va_win32
        vulkan-1
    )
    target_link_options(avs2dec_ts_cli PRIVATE
        "-Wl,--end-group"
    )
    set_target_properties(avs2dec_ts_cli PROPERTIES OUTPUT_NAME avs2dec_ts)
endif()

if(MSVC)
    target_compile_definitions(avs2dec PRIVATE _CRT_SECURE_NO_WARNINGS)
    if(AVS2DEC_BUILD_CLI)
        target_compile_definitions(avs2dec_cli PRIVATE _CRT_SECURE_NO_WARNINGS)
    endif()
    if(AVS2DEC_BUILD_CLI_TS)
        target_compile_definitions(avs2dec_ts_cli PRIVATE _CRT_SECURE_NO_WARNINGS)
    endif()
endif()
