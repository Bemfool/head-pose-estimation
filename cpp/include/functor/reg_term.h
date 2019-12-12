#include <cmath>
#include <ctime>
#include "bfm.h"
#include "vec.h"
#include "ceres/ceres.h"
#include "transform.h"
#define _USE_MATH_DEFINES

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
	const double u = 0.0005;
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
	const double u = 0.0005;
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
	const double u = 0.0001;
};