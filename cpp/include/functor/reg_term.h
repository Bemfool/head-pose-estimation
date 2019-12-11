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
        T sum = T(0);
        for(int i=0; i<N_ID_PC; i++)
            sum += shape_coef[i];
        for(int i=0; i<N_EXPR_PC; i++)
            sum += expr_coef[i];
        residuals[0] = T(coef) * sum;
		return true;
	}

	static ceres::CostFunction *create() {
		return (new ceres::AutoDiffCostFunction<total_reg_term,1, N_ID_PC, N_EXPR_PC>(
			new total_reg_term()));
	}
private:
	const double coef = 0.0;
};
