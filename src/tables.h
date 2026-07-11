#ifndef AVS2DEC_SRC_TABLES_H
#define AVS2DEC_SRC_TABLES_H

#include <stdint.h>
#include "levels.h"

/* Frame rate table (index 0..12 maps frame_rate_id-1) */
extern const float avs2_tab_frame_rate[13];

/* Luma 8-tap interpolation filters, [4][8] indexed by fractional (0..3) */
extern const int8_t avs2_intpl_filters[4][8];
/* Chroma 4-tap interpolation filters, [8][4] indexed by fractional (0..7) */
extern const int8_t avs2_intpl_filters_c[8][4];

/* DCT transform matrices */
extern const int16_t avs2_t4[4][4];
extern const int16_t avs2_t8[8][8];
extern const int16_t avs2_t16[16][16];
extern const int16_t avs2_t32[32][32];

/* 4x4 secondary transform (NSST) */
extern const int16_t avs2_2t_luma[16];
extern const int16_t avs2_2t_chroma[16];

/* QP tables */
extern const uint8_t avs2_qp_scale_cr[64];
extern const int16_t avs2_iq_shift[80];
extern const uint16_t avs2_iq_tab[80];

/* Scan tables */
extern const uint8_t avs2_scan_4x4[16][2];
extern const uint8_t avs2_scan_cg_8x8[4][2];
extern const uint8_t avs2_scan_cg_16x16[16][2];
extern const uint8_t avs2_scan_cg_32x32[64][2];
extern const uint8_t avs2_scan_cg_1x4[4][2];
extern const uint8_t avs2_scan_cg_4x1[4][2];
extern const uint8_t avs2_scan_cg_2x8[16][2];
extern const uint8_t avs2_scan_cg_8x2[16][2];

/* CG 扫描分派表 [tu_level][tu_split_type] */
extern const uint8_t (*const avs2_tab_scan_cg[4][4])[2];

extern const uint8_t avs2_b8xy_to_zigzag[8][8];

/* Deblock alpha/beta tables (indexed by QP) */
extern const uint16_t avs2_alpha_table[64];
extern const uint8_t avs2_beta_table[64];

/* ALF region table */
extern const uint8_t avs2_alf_region_table[16];

/* Weight-quant default matrices */
extern const uint8_t avs2_wqm_default_4x4[16];
extern const uint8_t avs2_wqm_default_8x8[64];
extern const int32_t avs2_wq_param_default[2][6];

/* DMH mode offsets */
extern const int8_t avs2_dmh_pos[DMH_MODE_NUM][2][2];

/* Chroma -> luma mode mapping */
extern const uint8_t avs2_tab_chroma_to_real[5];

/* AEC update tables */
extern const int8_t avs2_tab_cwr[8];
extern const int16_t avs2_tab_lg_pmps_offset[8];

/* Accessors for runtime-generated DCT matrices (t16/t32). */
const int16_t *avs2_get_t16(void);
const int16_t *avs2_get_t32(void);

#endif
