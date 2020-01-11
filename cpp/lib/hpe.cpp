#include "hpe.h"

hpe::hpe(std::string filename) {
	init(filename);
}

void hpe::init(std::string filename) {
	model.init(filename);
}

void hpe::estimate_ext_parm() {
	const double *int_parm = model.get_intrinsic_parm();
	double camera_vec[3][3] = {
		{int_parm[0], 0.0, int_parm[2]},
		{0.0, int_parm[1], int_parm[3]},
		{0.0, 0.0, 1.0}
	};
	cv::Mat camera_matrix = cv::Mat(3, 3, CV_64FC1, camera_vec);
	std::cout << "camera matrix: " << camera_matrix << std::endl;

	std::vector<float> dist_coef(0);
	std::vector<cv::Point3f> out;
	std::vector<cv::Point2f> in;
	cv::Mat rvec,tvec;
	dlib::matrix<double> fp_shape = model.get_fp_current_blendshape();
	for(int i=0; i<68; i++) {
		out.push_back(cv::Point3f(fp_shape(i*3), fp_shape(i*3+1), fp_shape(i*3+2)));
		in.push_back(cv::Point2f(observed_points.part(i).x(), observed_points.part(i).y()));
	}
	cv::solvePnP(out, in, camera_matrix, dist_coef, rvec, tvec);
	cv::Rodrigues(rvec, rvec);
	std::cout << rvec << std::endl;
	std::cout << tvec << std::endl;
	model.set_R(rvec);
	model.set_tx(tvec.at<double>(0));
	model.set_ty(tvec.at<double>(1));
	model.set_tz(tvec.at<double>(2));
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
	options.minimizer_progress_to_stdout = false;
	ceres::Solver::Summary summary;
	double u = 1.0f, v = 1.0;
	double u_step = 0.f, v_step = 0.f;

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
		std::cout << small_ext_parm[0] << " " << small_ext_parm[1] << " " << small_ext_parm[2] << " " 
		          << small_ext_parm[3] << " " << small_ext_parm[4] << " " << small_ext_parm[5] << " " << std::endl;	
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
