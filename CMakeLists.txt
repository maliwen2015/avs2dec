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

# SIMD 源文件仅使用 SSE4.1/SSSE3 intrinsics (AVX2 代码已全部清除)
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
        COMPILE_OPTIONS "/arch:SSE4.1"
    )
else()
    # GCC/MinGW: 仅启用 SSE4.1 + SSSE3 (无 AVX2)
    set_source_files_properties(${AVS2DEC_SIMD_SOURCES} PROPERTIES
        COMPILE_OPTIONS "-msse4.1;-mssse3"
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

if(MSVC)
    target_compile_definitions(avs2dec PRIVATE _CRT_SECURE_NO_WARNINGS)
    if(AVS2DEC_BUILD_CLI)
        target_compile_definitions(avs2dec_cli PRIVATE _CRT_SECURE_NO_WARNINGS)
    endif()
    if(AVS2DEC_BUILD_CLI_TS)
        target_compile_definitions(avs2dec_ts_cli PRIVATE _CRT_SECURE_NO_WARNINGS)
    endif()
endif()
