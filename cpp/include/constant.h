#pragma once
#include <string>

#define USE_HDF5

static std::string bfm_path = "../data/";

/* HDF5 Format Data */
static std::string bfm_h5_path = bfm_path + "model2017-1_bfm_nomouth.h5";

/* BFM 2017 */
#define N_VERTICE   53149
#define N_FACE      105694
#define N_ID_PC     199
#define N_EXPR_PC   100

#define FAIL	 -1
#define SUCCESS 0

/* data type (deprecated) */
typedef int d_type;
enum {
	BINARY_DATA = 0,	
	TEXT_DATA   = 1,
	FLOAT_DATA  = 2,
	DOUBLE_DATA = 3,
	VEC3_DATA   = 4,
	HDF5_DATA   = 5,
};

/* intrinsic parameters (from calibration) */
#define FX 1744.327628674942
#define FY 1747.838275588676
#define CX 800
#define CY 600

/* number of landmarks */
#define LANDMARK_NUM 68

/* file name of 3d standard model landmarks */
#define LANDMARK_FILE_NAME "landmarks.txt"

/* camera type */
typedef int camera_type;
enum {
	PARALLEL = 0,
	PINHOLE  = 1,
};

typedef int op_type;
enum {
	COARSE = 0,
	REAL   = 1,
};
