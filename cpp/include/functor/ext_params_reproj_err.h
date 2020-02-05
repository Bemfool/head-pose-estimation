#pragma once
#define _USE_MATH_DEFINES
#include <cmath>
#include <ctime>
#include "bfm.h"
#include "vec.h"
#include "ceres/ceres.h"
#include "transform.h"
#include "db_params.h"
#include "io_utils.h"
#include "type_utils.h"

class ext_params_reproj_err 
{
public:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
	ext_params_reproj_err(dlib::full_object_detection &observed_points_, bfm &model_) 
	: observed_points(observed_points_), model(model_) { }
	
    template<typename _Tp>
	bool operator () (const _Tp* const x, _Tp* residuals) const 
	{
		_Tp fx = _Tp(model.get_fx()), fy = _Tp(model.get_fy());
		_Tp cx = _Tp(model.get_cx()), cy = _Tp(model.get_cy());
		const dlib::matrix<double> _fp_shape = model.get_fp_current_blendshape();
		const dlib::matrix<_Tp> fp_shape = transform_points(x, _fp_shape);

		for(int i=0; i<N_LANDMARK; i++) 
		{
			_Tp u = fx * fp_shape(i*3) / fp_shape(i*3+2) + cx;
			_Tp v = fy * fp_shape(i*3+1) / fp_shape(i*3+2) + cy;
			residuals[i*2] = _Tp(observed_points.part(i).x()) - u;
			residuals[i*2+1] = _Tp(observed_points.part(i).y()) - v;
		}
		return true;
	}

	static ceres::CostFunction *create(dlib::full_object_detection &observed_points, bfm &model) 
	{
		return (new ceres::AutoDiffCostFunction<ext_params_reproj_err, N_LANDMARK * 2, N_EXT_PARAMS>(
			new ext_params_reproj_err(observed_points, model)));
	}

private:
	dlib::full_object_detection observed_points;
    bfm model;
};

class test_ext_params_reproj_err 
{
public:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
	test_ext_params_reproj_err(dlib::full_object_detection *observed_points, bfm *model) 
	: _observed_points(observed_points), _model(model) { }

	test_ext_params_reproj_err(dlib::full_object_detection *observed_points, bfm *model, double a, double b) 
	: _observed_points(observed_points), _model(model), _a(a), _b(b) {}


    template<typename _Tp>
	bool operator () (const _Tp* const x, _Tp* residuals) const 
	{
		_Tp fx = _Tp(_model->get_fx()), fy = _Tp(_model->get_fy());
		_Tp cx = _Tp(_model->get_cx()), cy = _Tp(_model->get_cy());

		const dlib::matrix<double> _fp_shape = _model->get_fp_current_blendshape_transformed();
		const dlib::matrix<_Tp> fp_shape = transform_points(x, _fp_shape, true);

		for(int i=0; i<N_LANDMARK; i++) 
		{
			_Tp u = fx * fp_shape(i*3) / fp_shape(i*3+2) + cx;
			_Tp v = fy * fp_shape(i*3+1) / fp_shape(i*3+2) + cy;
			residuals[i*2] = _Tp(_observed_points->part(i).x()) - u;
			residuals[i*2+1] = _Tp(_observed_points->part(i).y()) - v;	
		}

		/* regularialization */
	    residuals[N_LANDMARK*2]   = _Tp(_a) * x[0];
		residuals[N_LANDMARK*2+1] = _Tp(_a) * x[1];
		residuals[N_LANDMARK*2+2] = _Tp(_a) * x[2];
		residuals[N_LANDMARK*2+3] = _Tp(_b) * x[3];
		residuals[N_LANDMARK*2+4] = _Tp(_b) * x[4];
		residuals[N_LANDMARK*2+5] = _Tp(_b) * x[5];
		// print_array(residuals+N_LANDMARK*2, 6);
		
		return true;
	}

	static ceres::CostFunction *create(dlib::full_object_detection *observed_points, bfm *model) 
	{
		return (new ceres::AutoDiffCostFunction<test_ext_params_reproj_err, N_LANDMARK * 2 + N_EXT_PARAMS, N_EXT_PARAMS>(
			new test_ext_params_reproj_err(observed_points, model)));
	}

	static ceres::CostFunction *create(dlib::full_object_detection *observed_points, bfm *model, double a, double b) 
	{
		return (new ceres::AutoDiffCostFunction<test_ext_params_reproj_err, N_LANDMARK * 2 + N_EXT_PARAMS, N_EXT_PARAMS>(
			new test_ext_params_reproj_err(observed_points, model, a, b)));
	}


private:
	dlib::full_object_detection *_observed_points;
    bfm *_model;
	/* residual coefficients */
	double _a = 1.0;	/* for rotation */
	double _b = 1.0;	/* for translation */
};