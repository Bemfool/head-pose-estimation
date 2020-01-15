#pragma once
#include <cmath>
#include <ctime>
#include "bfm.h"
#include "vec.h"
#include "ceres/ceres.h"
#include "transform.h"
#define _USE_MATH_DEFINES

class ext_parm_reproj_err 
{
public:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
	ext_parm_reproj_err(dlib::full_object_detection &observed_points_, bfm &model_) 
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
		return (new ceres::AutoDiffCostFunction<ext_parm_reproj_err, N_LANDMARK * 2, N_EXT_PARM>(
			new ext_parm_reproj_err(observed_points, model)));
	}

private:
	dlib::full_object_detection observed_points;
    bfm model;
};

class test_ext_parm_reproj_err 
{
public:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
	test_ext_parm_reproj_err(dlib::full_object_detection *observed_points, bfm *model) 
	: _observed_points(observed_points), _model(model) { }

	test_ext_parm_reproj_err(dlib::full_object_detection *observed_points, bfm *model, double a, double b) 
	: _observed_points(observed_points), _model(model), _a(a), _b(b) {}


    template<typename _Tp>
	bool operator () (const _Tp* const x, _Tp* residuals) const 
	{
		_Tp fx = _Tp(_model->get_fx()), fy = _Tp(_model->get_fy());
		_Tp cx = _Tp(_model->get_cx()), cy = _Tp(_model->get_cy());

		// std::cout << "for R: " << x[0] << " " << x[1] << " " << x[2] << " " << x[3] << " " << x[4] << " " << x[5] << std::endl;
		const dlib::matrix<double> _fp_shape = _model->get_fp_current_blendshape_transformed();
		// std::cout << _fp_shape(3) << " " << _fp_shape(4) << " " << _fp_shape(5) << std::endl;

		const dlib::matrix<_Tp> fp_shape = transform_points(x, _fp_shape, true);
		// std::cout << fp_shape(3) << " " << fp_shape(4) << " " << fp_shape(5) << std::endl;

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
		// std::cout << "res: " << std::endl;
		// std::cout << residuals[68*2] << " " << residuals[68*2+1] << " " << residuals[68*2+2] << std::endl; 
		// std::cout << residuals[68*2+3] << " " << residuals[68*2+4] << " " << residuals[68*2+5] << std::endl; 

		return true;
	}

	static ceres::CostFunction *create(dlib::full_object_detection *observed_points, bfm *model) 
	{
		return (new ceres::AutoDiffCostFunction<test_ext_parm_reproj_err, N_LANDMARK * 2 + N_EXT_PARM, N_EXT_PARM>(
			new test_ext_parm_reproj_err(observed_points, model)));
	}

	static ceres::CostFunction *create(dlib::full_object_detection *observed_points, bfm *model, double a, double b) 
	{
		return (new ceres::AutoDiffCostFunction<test_ext_parm_reproj_err, N_LANDMARK * 2 + N_EXT_PARM, N_EXT_PARM>(
			new test_ext_parm_reproj_err(observed_points, model, a, b)));
	}


private:
	dlib::full_object_detection *_observed_points;
    bfm *_model;
	/* residual coefficients */
	double _a = 1.0;	/* for rotation */
	double _b = 1.0;	/* for translation */
};