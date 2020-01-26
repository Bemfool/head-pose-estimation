#include "hpe.h"

hpe::hpe(std::string filename) {
	init(filename);
}

void hpe::init(std::string filename) {
	model.init(filename);
}


bool hpe::solve_ext_params(long mode, double u, double v) {
	if(mode & USE_OPENCV)
	{
		const double *int_params = model.get_intrinsic_params();
		double camera_vec[3][3] = {
			{int_params[0], 0.0, int_params[2]},
			{0.0, int_params[1], int_params[3]},
			{0.0, 0.0, 1.0}
		};
		cv::Mat camera_matrix = cv::Mat(3, 3, CV_64FC1, camera_vec);
		std::cout << "camera matrix: " << camera_matrix << std::endl;

		std::vector<float> dist_coef(0);
		std::vector<cv::Point3f> out;
		std::vector<cv::Point2f> in;
		cv::Mat rvec, tvec;
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
		return true;
	}
	else if(mode & USE_LINEARIZED_RADIANS)
	{
		std::cout << "solve -> external parameters (linealized)" << std::endl;
		model.generate_transform_matrix();
		dlib::matrix<double> tmp =  model.get_fp_current_blendshape_transformed();
		tmp = dlib::reshape(tmp, tmp.nr() / 3, 3);
		mat_write("test.txt", tmp, "points");		

		std::cout << "init ceres solve - ";
		ceres::Solver::Options options;
		options.max_num_iterations = 100;
		options.num_threads = 8;
		options.minimizer_progress_to_stdout = false;
		ceres::Solver::Summary summary;
		double u_step = 0.f, v_step = 0.f;
		std::cout << "success" << std::endl;

		std::cout << "begin iteration" << std::endl;
		while(true) 
		{
			ceres::Problem problem;
			double small_ext_params[6] = { 0.f };
			ceres::CostFunction *cost_function = test_ext_params_reproj_err::create(&observed_points, &model, u, v);
			u = (u > u_step) ? u - u_step : 0.0;
			v = (v > v_step) ? v - v_step : 0.0;
			problem.AddResidualBlock(cost_function, nullptr, small_ext_params);
			ceres::Solve(options, &problem, &summary);
			std::cout << summary.BriefReport() << std::endl;

			if(is_close_enough(small_ext_params, 6)) 
			{
				print_array(small_ext_params, 6);
				std::cout << summary.BriefReport() << std::endl;
				break; 
			}

			model.accumulate_extrinsic_params(small_ext_params);
			print_array(small_ext_params, 6);
			mat_write("test.txt", model.get_R(), "R");
			mat_write("test.txt", model.get_T(), "T");
			dlib::matrix<double> tmp =  model.get_fp_current_blendshape_transformed();
			tmp = dlib::reshape(tmp, tmp.nr() / 3, 3);
			mat_write("test.txt", tmp, "points");

			// model.print_R();
			// model.print_T();
			// model.print_extrinsic_params();
			// std::cin.get();								
		}
		std::cout << "final u: " << u << " " << " v: " << v << std::endl;
		// model.generate_external_parameter();
		return (summary.termination_type == ceres::CONVERGENCE);
	}
	else
	{
		std::cout << "solve -> external parameters" << std::endl;	
		std::cout << "init ceres solve - ";
		ceres::Problem problem;
		double *ext_params = model.get_mutable_extrinsic_params();
		ceres::CostFunction *cost_function = ext_params_reproj_err::create(observed_points, model);
		problem.AddResidualBlock(cost_function, nullptr, ext_params);
		ceres::Solver::Options options;
		options.max_num_iterations = 100;
		options.num_threads = 8;
		options.minimizer_progress_to_stdout = true;
		ceres::Solver::Summary summary;
		std::cout << "success" << std::endl;
		ceres::Solve(options, &problem, &summary);
		std::cout << summary.BriefReport() << std::endl;
		model.generate_transform_matrix();
		return (summary.termination_type == ceres::CONVERGENCE);
	}
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


bool hpe::is_close_enough(double *ext_params, double rotation_eps, double translation_eps)
{
	for(int i=0; i<3; i++)
		if(abs(ext_params[i]) > rotation_eps)
			return false;
	for(int i=3; i<6; i++)
		if(abs(ext_params[i]) > translation_eps)
			return false;
	return true;
}

