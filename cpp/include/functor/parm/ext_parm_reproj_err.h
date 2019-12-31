#pragma once
#include <cmath>
#include <ctime>
#include "bfm.h"
#include "vec.h"
#include "ceres/ceres.h"
#include "transform.h"
#define _USE_MATH_DEFINES

class ext_parm_reproj_err {
public:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
	ext_parm_reproj_err(dlib::full_object_detection &observed_points_, bfm &model_) 
	: observed_points(observed_points_), model(model_) {}
	
    template<typename T>
	bool operator () (const T* const x, T* residuals) const {
		T fx = T(model.get_fx()), fy = T(model.get_fy());
		T cx = T(model.get_cx()), cy = T(model.get_cy());
		const dlib::matrix<double> _fp_shape = model.get_fp_current_blendshape();
		const dlib::matrix<T> fp_shape = transform(x, _fp_shape);

		for(int i=0; i<N_LANDMARK; i++) {
			T u = fx * fp_shape(i*3) / fp_shape(i*3+2) + cx;
			T v = fy * fp_shape(i*3+1) / fp_shape(i*3+2) + cy;
			residuals[i*2] = T(observed_points.part(i).x()) - u;
			residuals[i*2+1] = T(observed_points.part(i).y()) - v;		
		}
		return true;
	}

	static ceres::CostFunction *create(dlib::full_object_detection &observed_points, bfm &model) {
		return (new ceres::AutoDiffCostFunction<ext_parm_reproj_err, N_LANDMARK * 2, N_EXT_PARM>(
			new ext_parm_reproj_err(observed_points, model)));
	}

private:
	dlib::full_object_detection observed_points;
    bfm model;
};

class test_ext_parm_reproj_err {
public:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
	test_ext_parm_reproj_err(dlib::full_object_detection *observed_points_, bfm *model_) 
	: observed_points(observed_points_), model(model_) {}
	test_ext_parm_reproj_err(dlib::full_object_detection *observed_points_, bfm *model_, double u_, double v_) 
	: observed_points(observed_points_), model(model_), u(u_), v(v_) {}


    template<typename T>
	bool operator () (const T* const x, T* residuals) const {
		T fx = T(model->get_fx()), fy = T(model->get_fy());
		T cx = T(model->get_cx()), cy = T(model->get_cy());

		// std::cout << "for R: " << x[0] << " " << x[1] << " " << x[2] << " " << x[3] << " " << x[4] << " " << x[5] << std::endl;
		const dlib::matrix<double> _fp_shape = model->get_fp_current_blendshape_transformed();
		// std::cout << _fp_shape(3) << " " << _fp_shape(4) << " " << _fp_shape(5) << std::endl;

		const dlib::matrix<T> fp_shape = linearized_transform(x, _fp_shape);
		// std::cout << fp_shape(3) << " " << fp_shape(4) << " " << fp_shape(5) << std::endl;

		for(int i=0; i<N_LANDMARK; i++) {
			T u = fx * fp_shape(i*3) / fp_shape(i*3+2) + cx;
			T v = fy * fp_shape(i*3+1) / fp_shape(i*3+2) + cy;
			residuals[i*2] = T(observed_points->part(i).x()) - u;
			residuals[i*2+1] = T(observed_points->part(i).y()) - v;		
		}

	    residuals[N_LANDMARK*2]   = T(u) * x[0];
		residuals[N_LANDMARK*2+1] = T(u) * x[1];
		residuals[N_LANDMARK*2+2] = T(u) * x[2];
		residuals[N_LANDMARK*2+3] = T(v) * x[3];
		residuals[N_LANDMARK*2+4] = T(v) * x[4];
		residuals[N_LANDMARK*2+5] = T(v) * x[5];

		return true;
	}

	static ceres::CostFunction *create(dlib::full_object_detection *observed_points, bfm *model) {
		return (new ceres::AutoDiffCostFunction<test_ext_parm_reproj_err, N_LANDMARK * 2 + N_EXT_PARM, N_EXT_PARM>(
			new test_ext_parm_reproj_err(observed_points, model)));
	}
	static ceres::CostFunction *create(dlib::full_object_detection *observed_points, bfm *model, double u, double v) {
		return (new ceres::AutoDiffCostFunction<test_ext_parm_reproj_err, N_LANDMARK * 2 + N_EXT_PARM, N_EXT_PARM>(
			new test_ext_parm_reproj_err(observed_points, model, u, v)));
	}


private:
	dlib::full_object_detection *observed_points;
    bfm *model;
	double u = 1.0;
	double v = 1.0;
};