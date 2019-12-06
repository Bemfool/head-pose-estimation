#pragma once
#include "hpe.h"

struct HeadPoseNumericCostFunctor {
public:
	HeadPoseNumericCostFunctor(dlib::full_object_detection _shape);
	bool operator()(const double* const x, double* residual) const;
private:
	dlib::full_object_detection shape;	/* 3d landmarks coordinates got from dlib */
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