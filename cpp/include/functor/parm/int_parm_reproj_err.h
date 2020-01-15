#pragma once
#include <cmath>
#include <ctime>
#include "bfm.h"
#include "vec.h"
#include "ceres/ceres.h"
#include "transform.h"
#define _USE_MA_TpH_DEFINES

class int_parm_reproj_err {
public:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
	int_parm_reproj_err(dlib::full_object_detection &observed_points_, bfm &model_) 
	: observed_points(observed_points_), model(model_) {}
	
    template<typename _Tp>
	bool operator () (const _Tp* const x, _Tp* residuals) const {
		_Tp fx = x[0], fy = x[1];
		_Tp cx = x[2], cy = x[3];
		const dlib::matrix<double> _fp_shape = model.get_fp_current_blendshape();
        const double *ext_parm = model.get_external_parm();
        _Tp *ext_parm_ = new _Tp[6];
        for(int i=0; i<6; i++)
            ext_parm_[i] = (_Tp)(ext_parm[i]);
		const dlib::matrix<_Tp> fp_shape = transform_points(ext_parm_, _fp_shape);

		for(int i=0; i<N_LANDMARK; i++) {
			_Tp u = fx * fp_shape(i*3) / fp_shape(i*3+2) + cx;
			_Tp v = fy * fp_shape(i*3+1) / fp_shape(i*3+2) + cy;
			residuals[i*2] = _Tp(observed_points.part(i).x()) - u;
			residuals[i*2+1] = _Tp(observed_points.part(i).y()) - v;		
		}
		return true;
	}

	static ceres::CostFunction *create(dlib::full_object_detection &observed_points, bfm &model) {
		return (new ceres::AutoDiffCostFunction<int_parm_reproj_err, N_LANDMARK * 2, N_INT_PARM>(
			new int_parm_reproj_err(observed_points, model)));
	}

private:
	dlib::full_object_detection observed_points;
    bfm model;
};
