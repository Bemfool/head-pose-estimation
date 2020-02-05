#pragma once

/* just for my database */
const int N_LANDMARK = 68;
const int N_ID_PC = 99;
const int N_EXPR_PC = 29;
const int N_EXT_PARAMS = 6;
const int N_INT_PARAMS = 4;


enum solve_ext_parm_mode 
{
	USE_CERES  = 0L << 0,
	USE_OPENCV = 1L << 0,
	USE_LINEARIZED_RADIANS = 1L << 1,
};
