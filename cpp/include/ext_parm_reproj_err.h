#include "bfm.h"
#include <cmath>
#include "vec.h"
#include "ceres/ceres.h"
#define _USE_MATH_DEFINES

class ext_parm_reproj_err {
public:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
	ext_parm_reproj_err(dlib::full_object_detection &observed_points_, bfm &model_) 
	: observed_points(observed_points_), model(model_) {}
	
    template<typename T>
	bool operator                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               () (const T* const x, T* residuals) const {
		T yaw   = x[0];
		T pitch = x[1];
		T roll  = x[2];
		T tx    = x[3];
		T ty    = x[4];
		T tz    = x[5];

		T c1 = cos(roll  * T(M_PI) / T(180.0)), s1 = sin(roll  * T(M_PI) / T(180.0));
		T c2 = cos(yaw   * T(M_PI) / T(180.0)), s2 = sin(yaw   * T(M_PI) / T(180.0));
		T c3 = cos(pitch * T(M_PI) / T(180.0)), s3 = sin(pitch * T(M_PI) / T(180.0));

		T fx = T(model.fx()), fy = T(model.fy());
		T cx = T(model.cx()), cy = T(model.cy());
		dlib::matrix<double> fp_shape = model.get_fp_current_blendshape();

		for(int i=0; i<model.get_n_landmark(); i++) {
			T X = T(fp_shape(i*3)), Y = T(fp_shape(i*3+1)), Z = T(fp_shape(i*3+2)); 
			X = ( c1 * c2) * X + (c1 * s2 * s3 - c3 * s1) * Y + ( s1 * s3 + c1 * c3 * s2) * Z + tx;
			Y = ( c2 * s1) * X + (c1 * c3 + s1 * s2 * s3) * Y + (-c3 * s1 * s2 - c1 * s3) * Z + ty;
			Z = (-s2     ) * X + (c2 * s3               ) * Y + ( c2 * c3               ) * Z + tz; 
			T u = fx * X / Z + cx;
			T v = fy * Y / Z + cy;

			residuals[i*2] = T(observed_points.part(i).x()) - u;
			residuals[i*2+1] = T(observed_points.part(i).y()) - v;
		}
		return true;
	}

	static ceres::CostFunction *create(dlib::full_object_detection &observed_points, bfm &model) {
		return (new ceres::AutoDiffCostFunction<ext_parm_reproj_err, 68 * 2, 6>(
			new ext_parm_reproj_err(observed_points, model)));
	}

private:
	dlib::full_object_detection observed_points;
    bfm model;
};