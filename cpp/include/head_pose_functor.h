#pragma once
#include "hpe.h"

struct HeadPoseNumericCostFunctor {
public:
	HeadPoseNumericCostFunctor(dlib::full_object_detection _shape);
	bool operator()(const double* const x, double* residual) const;
private:
	dlib::full_object_detection shape;	/* 3d landmarks coordinates got from dlib */
};

class HeadPoseAutoDiffFunctor {
public:
	HeadPoseAutoDiffFunctor(point2d observed_point_, point3f model_point_) 
	: observed_point(observed_point_), model_point(model_point_) {}
	template<typename T>
	bool operator() (const T* const x, T* residuals) const {
		T yaw   = x[0];
		T pitch = x[1];
		T roll  = x[2];
		T tx    = x[3];
		T ty    = x[4];
		T tz    = x[5];
		T scale = x[6];

		/* We use Z1Y2X3 format of Tait–Bryan angles */
		T c1 = cos(roll  * T(pi) / T(180.0)), s1 = sin(roll  * T(pi) / T(180.0));
		T c2 = cos(yaw   * T(pi) / T(180.0)), s2 = sin(yaw   * T(pi) / T(180.0));
		T c3 = cos(pitch * T(pi) / T(180.0)), s3 = sin(pitch * T(pi) / T(180.0));

		T X = T(model_point.x()) * scale, Y = T(model_point.y()) * scale, Z = T(model_point.z()) * scale; 
		X = ( c1 * c2) * X + (c1 * s2 * s3 - c3 * s1) * Y + ( s1 * s3 + c1 * c3 * s2) * Z + tx;
		Y = ( c2 * s1) * X + (c1 * c3 + s1 * s2 * s3) * Y + (-c3 * s1 * s2 - c1 * s3) * Z + ty;
		Z = (-s2     ) * X + (c2 * s3               ) * Y + ( c2 * c3               ) * Z + tz; 
		T u = T(FX) * X / Z + T(CX);
		T v = T(FY) * Y / Z + T(CY);

		residuals[0] = T(observed_point.x()) - u;
		residuals[1] = T(observed_point.y()) - v;
		return true;
	}

	static ceres::CostFunction *create(point2d observed_point, point3f model_point) {
		return (new ceres::AutoDiffCostFunction<HeadPoseAutoDiffFunctor, 2, 7>(
			new HeadPoseAutoDiffFunctor(observed_point, model_point)));
	}

private:
	point2d observed_point;
	point3f model_point;
};

void solve_head_pose(dlib::full_object_detection &observed_points, double *head_pose) {
	ceres::Problem problem;
	for(int i=0; i<LANDMARK_NUM; i++) {
		ceres::CostFunction *cost_function = HeadPoseAutoDiffFunctor::create(observed_points.part(i), get_landmark(i));
		ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
		problem.AddResidualBlock(cost_function, loss_function, head_pose);
	}
	ceres::Solver::Options options;
	options.minimizer_progress_to_stdout = true;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.BriefReport() << std::endl;
}