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
        # 注: 不使用 -march=native, 避免生成的二进制在较旧 CPU 上因非法指令崩溃.
        # SIMD 源文件通过 -msse4.1 -mssse3 单独启用所需指令集.
        add_compile_options(-O3 -flto -funroll-loops)
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
# tables.c 使用 cos(); GCC/Clang 需链接 libm (MSVC 数学库在 UCRT 中)
if(NOT MSVC)
    target_link_libraries(avs2dec PRIVATE m)
endif()

# SIMD 源文件主体使用 SSE4.1/SSSE3 intrinsics 编译;
# AVX2 内核 (mc_simd.c) 通过 __attribute__((target("avx2"))) 局部启用,
# 由运行时 CPU 检测分发, 不需要全局 -mavx2 标志.
# simd_init.c 不含 intrinsics, 不需要特殊标志
# cpu.c 需要 -march=armv8.2-a+dotprod: 其 dotprod 运行时检测被
# __ARM_FEATURE_DOTPROD 宏包裹, 无该标志时检测代码被排除, SDOT 永远不启用
set(AVS2DEC_SIMD_SOURCES
    src/cpu.c
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
    # GCC/MinGW/Clang: 按目标架构启用对应 SIMD 指令集.
    # x86: SSE4.1 + SSSE3 (AVX2 内核用 target 属性局部启用, 见上方说明);
    # aarch64: NEON 默认启用, 无需标志; arm32: -mfpu=neon.
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|amd64|AMD64|i386|i686|X86|x86)")
        set_source_files_properties(${AVS2DEC_SIMD_SOURCES} PROPERTIES
            COMPILE_OPTIONS "-msse4.1;-mssse3"
        )
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64|arm64|ARM64)")
        # aarch64: NEON 为默认. 启用 ARMv8.2 dotprod (SDOT) 以加速 16x16/32x32 IDCT.
        # 运行时通过 HWCAP_ASIMDDP 检测, 不支持 dotprod 的 CPU 自动回退蝶形 NEON.
        set_source_files_properties(${AVS2DEC_SIMD_SOURCES} PROPERTIES
            COMPILE_OPTIONS "-march=armv8.2-a+dotprod"
        )
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm|ARM)")
        set_source_files_properties(${AVS2DEC_SIMD_SOURCES} PROPERTIES
            COMPILE_OPTIONS "-mfpu=neon"
        )
    endif()
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
endif()
