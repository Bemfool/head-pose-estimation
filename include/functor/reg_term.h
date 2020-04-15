#pragma once
#define _USE_MATH_DEFINES
#include <cmath>
#include <ctime>
#include "bfm.h"
#include "vec.h"
#include "ceres/ceres.h"
#include "transform.h"
#include "db_params.h"

class total_reg_term {
public:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
	total_reg_term() {}
	
    template<typename T>
	bool operator () (const T* const shape_coef, const T* const expr_coef, T* residuals) const {
        for(int i=0; i<N_ID_PC; i++)
            residuals[i] = T(u) * shape_coef[i];
        for(int i=0; i<N_EXPR_PC; i++)
            residuals[i + N_ID_PC] = T(v) * expr_coef[i];
		return true;
	}

	static ceres::CostFunction *create() {
		return (new ceres::AutoDiffCostFunction<total_reg_term, N_ID_PC + N_EXPR_PC, N_ID_PC, N_EXPR_PC>(
			new total_reg_term()));
	}
private:
	const double u = 0.1;
	const double v = 1.0;
};


class shape_coef_reg_term {
public:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
	shape_coef_reg_term() {}
    template<typename T>
	bool operator () (const T* const shape_coef, T* residuals) const {
        for(int i=0; i<N_ID_PC; i++)
	        residuals[i] = T(u) * shape_coef[i];
		return true;
	}

	static ceres::CostFunction *create() {
		return (new ceres::AutoDiffCostFunction<shape_coef_reg_term, N_ID_PC, N_ID_PC>(
			new shape_coef_reg_term()));
	}
private:
	const double u = 0.003;
};

class expr_coef_reg_term {
public:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
	expr_coef_reg_term() {}
    template<typename T>
	bool operator () (const T* const expr_coef, T* residuals) const {
        for(int i=0; i<N_EXPR_PC; i++)
	        residuals[i] = T(u) * expr_coef[i];
		return true;
	}

	static ceres::CostFunction *create() {
		return (new ceres::AutoDiffCostFunction<expr_coef_reg_term, N_EXPR_PC, N_EXPR_PC>(
			new expr_coef_reg_term()));
	}
private:
	const double u = 0.01;
};


class ext_parm_reg_term {
public:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
	ext_parm_reg_term() {}
    template<typename T>
	bool operator () (const T* const ext_parm, T* residuals) const {
	    residuals[0] = T(u) * ext_parm[0];
		residuals[1] = T(u) * ext_parm[1];
		residuals[2] = T(u) * (ext_parm[2] + 180.0);
	    residuals[3] = T(v) * ext_parm[3];
		residuals[4] = T(v) * ext_parm[4];
		residuals[5] = T(v) * ext_parm[5];
		return true;
	}

	static ceres::CostFunction *create() {
		return (new ceres::AutoDiffCostFunction<ext_parm_reg_term, 6, 6>(
			new ext_parm_reg_term()));
	}
private:
	const double u = 3.0;
	const double v = 0.001;
};
