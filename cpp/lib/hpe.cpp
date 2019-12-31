#include "hpe.h"

hpe::hpe(std::string filename) {
	init(filename);
}

void hpe::init(std::string filename) {
	model.init(filename);
}

void hpe::estimate_ext_parm() {

}


bool hpe::solve_total() {
	estimate_ext_parm();
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
	return (summary.termination_type == ceres::CONVERGENCE);
}

bool hpe::solve_parm() {
	ceres::Problem problem;
	double *ext_parm = model.get_mutable_external_parm();
	double *int_parm = model.get_mutable_intrinsic_parm();
	ceres::CostFunction *cost_function = parm_reproj_err::create(observed_points, model);
	problem.AddResidualBlock(cost_function, nullptr, ext_parm, int_parm);
	ceres::Solver::Options options;
	options.max_num_iterations = 100;
	options.num_threads = 8;
	options.minimizer_progress_to_stdout = true;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.BriefReport() << std::endl;	
	return (summary.termination_type == ceres::CONVERGENCE);
}



bool hpe::solve_ext_parm() {
	ceres::Problem problem;
	double *ext_parm = model.get_mutable_external_parm();
	ceres::CostFunction *cost_function = ext_parm_reproj_err::create(observed_points, model);
	problem.AddResidualBlock(cost_function, nullptr, ext_parm);
	ceres::Solver::Options options;
	options.max_num_iterations = 100;
	options.num_threads = 8;
	options.minimizer_progress_to_stdout = true;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.BriefReport() << std::endl;
	return (summary.termination_type == ceres::CONVERGENCE);
}


bool hpe::solve_ext_parm_test() {
	model.generate_rotation_matrix();

	ceres::Solver::Options options;
	options.max_num_iterations = 100;
	options.num_threads = 8;
	options.minimizer_progress_to_stdout = true;
	ceres::Solver::Summary summary;
	double u = 1.0, v = 0.1;
	double u_step = 0.000001, v_step = 0.000001;

	while(true) {
		ceres::Problem problem;
		double small_ext_parm[6] = { 0.f };
		ceres::CostFunction *cost_function = test_ext_parm_reproj_err::create(&observed_points, &model, u, v);
		u = (u > u_step) ? u - u_step : 0.0;
		v = (v > v_step) ? v - v_step : 0.0;
		problem.AddResidualBlock(cost_function, nullptr, small_ext_parm);
		ceres::Solve(options, &problem, &summary);
		// std::cout << summary.BriefReport() << std::endl;

		if(small_ext_parm[0] == 0 && small_ext_parm[1] == 0 &&
		   small_ext_parm[2] == 0 && small_ext_parm[3] == 0 &&
		   small_ext_parm[4] == 0 && small_ext_parm[5] == 0) {
		   std::cout << summary.BriefReport() << std::endl;
		   break; 
		}

		model.accumulate_external_parm(small_ext_parm);
		// std::cout << small_ext_parm[0] << " " << small_ext_parm[1] << " " << small_ext_parm[2] << " " 
		//           << small_ext_parm[3] << " " << small_ext_parm[4] << " " << small_ext_parm[5] << " " << std::endl;	
		// model.print_R();
		// model.print_external_parm();

		// std::cin.get();								
	}
	std::cout << "final u: " << u << " " << " v: " << v << std::endl;
	return (summary.termination_type == ceres::CONVERGENCE);
}


bool hpe::solve_int_parm() {
	ceres::Problem problem;
	double *int_parm = model.get_mutable_intrinsic_parm();
	ceres::CostFunction *cost_function = int_parm_reproj_err::create(observed_points, model);
	// ceres::CostFunction *reg_function = ext_parm_reg_term::create();
	problem.AddResidualBlock(cost_function, nullptr, int_parm);
	// problem.AddResidualBlock(reg_function, nullptr, ext_parm);
	ceres::Solver::Options options;
	options.max_num_iterations = 100;
	options.num_threads = 8;
	options.minimizer_progress_to_stdout = true;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.BriefReport() << std::endl;
	return (summary.termination_type == ceres::CONVERGENCE);
}


bool hpe::solve_shape_coef() {
	ceres::Problem problem;
	double *shape_coef = model.get_mutable_shape_coef();
	ceres::CostFunction *cost_function = shape_coef_reproj_err::create(observed_points, model);
	ceres::CostFunction *reg_term = shape_coef_reg_term::create();
	problem.AddResidualBlock(cost_function, nullptr, shape_coef);
	problem.AddResidualBlock(reg_term, nullptr, shape_coef);
	ceres::Solver::Options options;
	options.max_num_consecutive_invalid_steps = 10;
	options.num_threads = 8;
	options.minimizer_progress_to_stdout = true;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.BriefReport() << std::endl;
	model.generate_fp_face();
	return (summary.termination_type == ceres::CONVERGENCE);
}


bool hpe::solve_expr_coef() {
	ceres::Problem problem;
	double *expr_coef = model.get_mutable_expr_coef();
	ceres::CostFunction *cost_function = expr_coef_reproj_err::create(observed_points, model);
	ceres::CostFunction *reg_term = expr_coef_reg_term::create();
	problem.AddResidualBlock(cost_function, nullptr, expr_coef);
	problem.AddResidualBlock(reg_term, nullptr, expr_coef);
	ceres::Solver::Options options;
	options.max_num_consecutive_invalid_steps = 10;
	options.num_threads = 8;
	options.minimizer_progress_to_stdout = true;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.BriefReport() << std::endl;
	model.generate_fp_face();
	return (summary.termination_type == ceres::CONVERGENCE);
}
