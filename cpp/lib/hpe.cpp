#include "hpe.h"

hpe::hpe(std::string filename) {
	init(filename);
}

void hpe::init(std::string filename) {
	model.init(filename);
}


void hpe::solve_total() {
	ceres::Problem problem;
	double *ext_parm = model.get_mutable_external_parm();
	double *shape_coef = model.get_mutable_shape_coef();
	double *expr_coef = model.get_mutable_expr_coef();
	ceres::CostFunction *cost_function = total_reproj_err::create(observed_points, model);
	ceres::CostFunction *reg_function = total_reg_term::create();
	problem.AddResidualBlock(cost_function, nullptr, ext_parm, shape_coef, expr_coef);
	problem.AddResidualBlock(reg_function, nullptr, shape_coef, expr_coef);
	ceres::Solver::Options options;
	options.num_threads = 8;
	options.minimizer_progress_to_stdout = true;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.BriefReport() << std::endl;
	model.generate_face();
}

void hpe::solve_ext_parm() {
	ceres::Problem problem;
	double *ext_parm = model.get_mutable_external_parm();
	ceres::CostFunction *cost_function = ext_parm_reproj_err::create(observed_points, model);
	problem.AddResidualBlock(cost_function, nullptr, ext_parm);
	ceres::Solver::Options options;
	options.num_threads = 8;
	options.minimizer_progress_to_stdout = true;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.BriefReport() << std::endl;
}

void hpe::solve_shape_coef() {
	ceres::Problem problem;
	double *shape_coef = model.get_mutable_shape_coef();
	ceres::CostFunction *cost_function = shape_coef_reproj_err::create(observed_points, model);
	problem.AddResidualBlock(cost_function, nullptr, shape_coef);
	ceres::Solver::Options options;
	options.max_num_consecutive_invalid_steps = 10;
	options.num_threads = 8;
	options.minimizer_progress_to_stdout = true;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.BriefReport() << std::endl;
}