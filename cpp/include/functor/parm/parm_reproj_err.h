#pragma once
#include <cmath>
#include <ctime>
#include "bfm.h"
#include "vec.h"
#include "ceres/ceres.h"
#include "transform.h"
#define _USE_MATH_DEFINES

class parm_reproj_err {
public:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
	parm_reproj_err(dlib::full_object_detection &observed_points_, bfm &model_) 
	: observed_points(observed_points_), model(model_) {}
	
    template<typename T>
	bool operator () (const T* const ext_parm, const T* const int_parm, T* residuals) const {
		T fx = int_parm[0], fy = int_parm[1];
		T cx = int_parm[2], cy = int_parm[3];
		const dlib::matrix<double> _fp_shape = model.get_fp_current_blendshape();
		T tmp[6] = { T(0) };
		tmp[0] = ext_parm[0];
		tmp[1] = ext_parm[1];
		tmp[2] = ext_parm[2];
		
		const dlib::matrix<T> fp_shape = transform(ext_parm, _fp_shape);

		for(int i=0; i<N_LANDMARK; i++) {
			T u = fx * fp_shape(i*3) / fp_shape(i*3+2) + cx;
			T v = fy * fp_shape(i*3+1) / fp_shape(i*3+2) + cy;
			residuals[i*2] = T(observed_points.part(i).x()) - u;
			residuals[i*2+1] = T(observed_points.part(i).y()) - v;		
		}
		return true;
	}

	static ceres::CostFunction *create(dlib::full_object_detection &observed_points, bfm &model) {
		return (new ceres::AutoDiffCostFunction<parm_reproj_err, N_LANDMARK * 2, 6, 4>(
			new parm_reproj_err(observed_points, model)));
	}

private:
	dlib::full_object_detection observed_points;
    bfm model;
};