#ifndef AVS2DEC_SRC_LEVELS_H
#define AVS2DEC_SRC_LEVELS_H

#include <stdint.h>

/*
 * AVS2 (GB/T 33475.2) constants and enumerations.
 * Naming follows dav1d style (snake_case) per project convention.
 */

/* ---- Profiles ---- */
#define AVS2_PROFILE_MAIN_PIC  0x12
#define AVS2_PROFILE_MAIN      0x20
#define AVS2_PROFILE_MAIN10    0x22

/* ---- Start codes ---- */
#define AVS2_SC_SEQUENCE_HEADER 0xB0
#define AVS2_SC_SEQUENCE_END    0xB1
#define AVS2_SC_USER_DATA       0xB2
#define AVS2_SC_INTRA_PICTURE   0xB3
#define AVS2_SC_EXTENSION       0xB5
#define AVS2_SC_INTER_PICTURE   0xB6
#define AVS2_SC_VIDEO_EDIT      0xB7
#define AVS2_SC_SLICE_MIN       0x00
#define AVS2_SC_SLICE_MAX       0x8F

/* ---- Slice/picture types ---- */
enum avs2_slice_type {
    AVS2_I_SLICE = 0,
    AVS2_P_SLICE = 1,
    AVS2_B_SLICE = 2,
    AVS2_G_SLICE = 3, /* GB background */
    AVS2_F_SLICE = 4,
    AVS2_S_SLICE = 5,
    AVS2_GB_SLICE = 6,
};

#define IS_INTRA(s) ((s) <= AVS2_G_SLICE && ((s) == AVS2_I_SLICE || (s) == AVS2_G_SLICE || (s) == AVS2_GB_SLICE))
#define IS_INTER(s) (!IS_INTRA(s))

/* ---- Prediction direction ---- */
enum avs2_pdir {
    PDIR_FWD  = 0,
    PDIR_BWD  = 1,
    PDIR_SYM  = 2,
    PDIR_BID  = 3,
    PDIR_DUAL = 4,
    PDIR_INVALID = -1,
};

/* B-frame reference slot order (BWD first) */
#define B_BWD 0
#define B_FWD 1

/* ---- CU/PU/TU sizes ---- */
#define MIN_CU_SIZE_IN_BIT 3
#define MIN_CU_SIZE        8
#define MAX_CU_SIZE_IN_BIT 6
#define MAX_CU_SIZE        64
#define MIN_PU_SIZE_IN_BIT 2
#define MIN_PU_SIZE        4
#define BLOCK_MULTIPLE     2  /* 1 CU = 2x2 PU (4x4 units) */

#define B4X4_IN_BIT  2
#define B8X8_IN_BIT  3
#define B16X16_IN_BIT 4
#define B64X64_IN_BIT 6

#define AVS2_GOP_NUM     32
#define AVS2_COI_CYCLE   256
#define TEMPORAL_MAXLEVEL_BIT 3
#define AVS2_MAX_REFS    4
#define MAX_POC_DISTANCE 128

/* ---- Coefficients ---- */
#define CG_SIZE        16   /* coefficient group 4x4 */
#define SEC_TR_SIZE    4
#define MAX_QP         63
#define LIMIT_BIT      16

/* ---- MV constants (1/4 pixel luma, 1/8 chroma) ---- */
#define MULTI         16384
#define HALF_MULTI    8192
#define OFFSET        14
#define THRESHOLD_PMVR 2
#define MV_FACTOR_IN_BIT 2
#define AVS2_DISTANCE_INDEX(distance)    (((distance) + 512) & 511)

/* ---- ALF ---- */
#define ALF_MAX_NUM_COEF 9
#define ALF_NUM_VARS     16
#define ALF_NUM_BIT_SHIFT 6
#define ALF_FOOTPRINT_SIZE 7

/* ---- Intra prediction modes (luma) ---- */
enum avs2_intra_luma_mode {
    DC_PRED      = 0,
    PLANE_PRED   = 1,
    BI_PRED      = 2,
    /* angular horizontal 3..12 */
    INTRA_ANG_X_3  = 3,  INTRA_ANG_X_4  = 4,  INTRA_ANG_X_5  = 5,
    INTRA_ANG_X_6  = 6,  INTRA_ANG_X_7  = 7,  INTRA_ANG_X_8  = 8,
    INTRA_ANG_X_9  = 9,  INTRA_ANG_X_10 = 10, INTRA_ANG_X_11 = 11,
    VERT_PRED      = 12,
    /* angular diagonal 13..23 */
    INTRA_ANG_XY_13 = 13, INTRA_ANG_XY_14 = 14, INTRA_ANG_XY_15 = 15,
    INTRA_ANG_XY_16 = 16, INTRA_ANG_XY_17 = 17, INTRA_ANG_XY_18 = 18,
    INTRA_ANG_XY_19 = 19, INTRA_ANG_XY_20 = 20, INTRA_ANG_XY_21 = 21,
    INTRA_ANG_XY_22 = 22, INTRA_ANG_XY_23 = 23,
    /* angular vertical 24..32 */
    HOR_PRED        = 24,
    INTRA_ANG_Y_25  = 25, INTRA_ANG_Y_26 = 26, INTRA_ANG_Y_27 = 27,
    INTRA_ANG_Y_28  = 28, INTRA_ANG_Y_29 = 29, INTRA_ANG_Y_30 = 30,
    INTRA_ANG_Y_31  = 31, INTRA_ANG_Y_32 = 32,
    NUM_INTRA_MODE  = 33,
};

/* ---- Intra chroma modes ---- */
enum avs2_intra_chroma_mode {
    DM_PRED_C  = 0,
    DC_PRED_C  = 1,
    HOR_PRED_C = 2,
    VERT_PRED_C = 3,
    BI_PRED_C  = 4,
    NUM_INTRA_MODE_C = 5,
};

/* ---- PU partition modes ---- */
enum avs2_pu_mode {
    PRED_SKIP    = 0,
    PRED_2Nx2N   = 1,
    PRED_2NxN    = 2,
    PRED_Nx2N    = 3,
    PRED_2NxnU   = 4,
    PRED_2NxnD   = 5,
    PRED_nLx2N   = 6,
    PRED_nRx2N   = 7,
    PRED_I_2Nx2N = 8,
    PRED_I_NxN   = 9,
    PRED_I_2Nxn  = 10, /* SDIP horizontal */
    PRED_I_nx2N  = 11, /* SDIP vertical */
};

/* ---- TU split ---- */
enum avs2_tu_split {
    TU_SPLIT_NON   = 0,
    TU_SPLIT_HOR   = 1,
    TU_SPLIT_VER   = 2,
    TU_SPLIT_CROSS = 3,
};

/* ---- SAO ---- */
enum avs2_sao_type {
    SAO_NOT_APPLIED = 0,
    SAO_BO          = 1,
    SAO_EO_HORZ     = 2,
    SAO_EO_VERT     = 3,
    SAO_EO_135      = 4,
    SAO_EO_45       = 5,
};

/* ---- DMH (Directional Multi-Hypothesis) positions ---- */
#define DMH_MODE_NUM 9

/* ---- Edge directions for loop filter ---- */
enum {
    EDGE_VER = 0,
    EDGE_HOR = 1,
};

#endif /* AVS2DEC_SRC_LEVELS_H */
