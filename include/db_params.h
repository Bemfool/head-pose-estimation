#pragma once

/* TODO
 * Update these parameters with your database 
 */
const int N_LANDMARK = 68;
// const int N_LANDMARK = 6;
// const int N_LANDMARK = 12;
const int N_ID_PC = 99;
const int N_EXPR_PC = 29;
const int N_EXT_PARAMS = 6;
const int N_INT_PARAMS = 4;

const int N_CERES_ITERATIONS = 100;
const int N_CERES_THREADS = 16;
#ifndef _DEBUG
const bool B_CERES_STDCOUT = true;
#else
const bool B_CERES_STDCOUT = false;
#endif

enum SolveExtParamsMode
{
	SolveExtParamsMode_InvalidFirst = 0L << 0,
	SolveExtParamsMode_UseCeres = 1L << 0,
	SolveExtParamsMode_UseOpenCV = 1L << 1,
	SolveExtParamsMode_UseLinearizedRadians = 1L << 2,
	SolveExtParamsMode_UseDlt = 1L << 3,
};
