#pragma once
#define _USE_MATH_DEFINES
#include <cmath>
#include <ctime>
#include "bfm.h"
#include "vec.h"
#include "ceres/ceres.h"
#include "transform.h"
#include "db_params.h"

class expr_coef_reproj_err {
public:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
	expr_coef_reproj_err(dlib::full_object_detection &observed_points_, bfm &model_) 
	: observed_points(observed_points_), model(model_) { }
	
    template<typename T>
	bool operator () (const T* const expr_coef, T* residuals) const {
		T fx = T(model.get_fx()), fy = T(model.get_fy());
		T cx = T(model.get_cx()), cy = T(model.get_cy());

		const dlib::matrix<T> fp_shape_ = model.generate_fp_face_by_expr(expr_coef);  

		const double *extrinsic_params = model.get_extrinsic_params();
        T *extrinsic_params_ = new T[6];
        for(int i=0; i<6; i++)
            extrinsic_params_[i] = (T)(extrinsic_params[i]);

		const dlib::matrix<T> fp_shape = transform_points(extrinsic_params_, fp_shape_);

		for(int i=0; i<N_LANDMARK; i++) {
			T u = fx * fp_shape(i*3) / fp_shape(i*3+2) + cx;
			T v = fy * fp_shape(i*3+1) / fp_shape(i*3+2) + cy;
			residuals[i*2] = T(observed_points.part(i).x()) - u;
			residuals[i*2+1] = T(observed_points.part(i).y()) - v;		
		}
		return true;
	}

	static ceres::CostFunction *create(dlib::full_object_detection &observed_points, bfm &model) {
		return (new ceres::AutoDiffCostFunction<expr_coef_reproj_err, N_LANDMARK * 2, N_EXPR_PC>(
			new expr_coef_reproj_err(observed_points, model)));
	}

private:
	dlib::full_object_detection observed_points;
	bfm model;
};
