#include "hpe.h"

hpe::hpe(std::string filename) {
	init(filename);
}

void hpe::init(std::string filename) {
	model.init(filename);
}

void hpe::dlt()
{
	cv::Ptr<CvMat> matL;
	double* L;
	double LL[12*12], LW[12], LV[12*12];
	CvMat _LL = cvMat( 12, 12, CV_64F, LL );
	CvMat _LW = cvMat( 12, 1, CV_64F, LW );
	CvMat _LV = cvMat( 12, 12, CV_64F, LV );
	CvMat _RRt, _RR, _tt;

	matL.reset(cvCreateMat( 2*N_LANDMARK, 12, CV_64F ));
	L = matL->data.db;
	dlib::matrix<double> fp_model = model.get_fp_current_blendshape();
	const double fx = model.get_fx();
	const double fy = model.get_fy();
	const double cx = model.get_cx();
	const double cy = model.get_cy();

	for(int i=0; i<N_LANDMARK; i++, L+=24)
	{
		double x = observed_points.part(i).x(), y = observed_points.part(i).y();
		double X = fp_model(i*3), Y = fp_model(i*3+1), Z = fp_model(i*3+2);
		L[0] = X * fx; L[16] = X * fy;
		L[1] = Y * fx; L[17] = Y * fy;
		L[2] = Z * fx; L[18] = Z * fy;
		L[3] = fx; L[19] = fy;
		L[4] = L[5] = L[6] = L[7] = 0.;
		L[12] = L[13] = L[14] = L[15] = 0.;
		L[8] = X * cx - x * X;
		L[9] = Y * cx - x * Y;
		L[10] = Z * cx - x * Z;
		L[11] = cx - x;
		L[20] = X * cy - y * X;
		L[21] = Y * cy - y * Y;
		L[22] = Z * cy - y * Z;
		L[23] = cy - y;
	}

	cvMulTransposed( matL, &_LL, 1 );
	cvSVD( &_LL, &_LW, 0, &_LV, CV_SVD_MODIFY_A + CV_SVD_V_T );
	_RRt = cvMat( 3, 4, CV_64F, LV + 11*12 );
	cvGetCols( &_RRt, &_RR, 0, 3 );
	cvGetCol( &_RRt, &_tt, 3 );
	model.set_R(&_RR);
	model.set_T(&_tt);
	model.generate_external_parameter();
}


bool hpe::solve_ext_params(long mode, double ca, double cb) 
{
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
		model.set_T(tvec);
		model.generate_external_parameter();
		return true;
	}
	else if(mode & USE_LINEARIZED_RADIANS)
	{
		std::cout << "solve -> external parameters (linealized)" << std::endl;
		std::cout << "	1) esitimate initial values by using DLT algorithm." << std::endl;
		dlt();
		// return true;

		model.generate_transform_matrix();

		// const double *int_parms = model.get_intrinsic_params();
		// dlib::matrix<double> tmp =  model.get_fp_current_blendshape_transformed();
		// tmp = dlib::reshape(tmp, tmp.nr() / 3, 3);
		// mat_write("test.txt", tmp, "points");		

		std::cout << "	2) iteration." << std::endl;
		std::cout << "init ceres solve - ";
		ceres::Solver::Options options;
		options.max_num_iterations = 100;
		options.num_threads = 8;
		options.minimizer_progress_to_stdout = false;
		ceres::Solver::Summary summary;
		std::cout << "success" << std::endl;

		std::cout << "begin iteration" << std::endl;
		while(true) 
		{
			ceres::Problem problem;
			double small_ext_params[6] = { 0.f };
			ceres::CostFunction *cost_function = test_ext_params_reproj_err::create(&observed_points, &model, ca, cb);
			problem.AddResidualBlock(cost_function, nullptr, small_ext_params);
			ceres::Solve(options, &problem, &summary);
			std::cout << summary.BriefReport() << std::endl;

			if(is_close_enough(small_ext_params, 0, 0)) 
			{
				print_array(small_ext_params, 6);
				std::cout << summary.BriefReport() << std::endl;
				break; 
			}

			// mat_write("test.txt", model.get_R(), "bR");
			// mat_write("test.txt", model.get_T(), "bT");
			// dlib::matrix<double> tmp =  model.get_fp_current_blendshape_transformed();
			// double sum = 0;
			// for(int i=0; i<N_LANDMARK; i++) 
			// {
			// 	double u = int_parms[0] * tmp(i*3) / tmp(i*3+2) + int_parms[2];
			// 	double v = int_parms[1] * tmp(i*3+1) / tmp(i*3+2) + int_parms[3];
			// 	sum = sum + (observed_points.part(i).x() - u) * (observed_points.part(i).x() - u);
			// 	sum = sum + (observed_points.part(i).y() - v) * (observed_points.part(i).y() - v);		
			// }
			// str_write("test.txt", to_string(sum/2));
			// const dlib::matrix<double> tmp2 = transform_points(small_ext_params, tmp, true);			
			// sum = 0;
			// for(int i=0; i<N_LANDMARK; i++) 
			// {
			// 	double u = int_parms[0] * tmp2(i*3) / tmp2(i*3+2) + int_parms[2];
			// 	double v = int_parms[1] * tmp2(i*3+1) / tmp2(i*3+2) + int_parms[3];
			// 	sum = sum + (observed_points.part(i).x() - u) * (observed_points.part(i).x() - u);
			// 	sum = sum + (observed_points.part(i).y() - v) * (observed_points.part(i).y() - v);		
			// }
			// str_write("test.txt", to_string(sum/2));

			model.accumulate_extrinsic_params(small_ext_params);
			print_array(small_ext_params, 6);

			// mat_write("test.txt", model.get_R(), "R");
			// mat_write("test.txt", model.get_T(), "T");
			// tmp =  model.get_fp_current_blendshape_transformed();
			// sum = 0;
			// for(int i=0; i<N_LANDMARK; i++) 
			// {
			// 	double u = int_parms[0] * tmp(i*3) / tmp(i*3+2) + int_parms[2];
			// 	double v = int_parms[1] * tmp(i*3+1) / tmp(i*3+2) + int_parms[3];
			// 	sum = sum + (observed_points.part(i).x() - u) * (observed_points.part(i).x() - u);
			// 	sum = sum + (observed_points.part(i).y() - v) * (observed_points.part(i).y() - v);		
			// }
			// str_write("test.txt", to_string(sum/2));
			// tmp = dlib::reshape(tmp, tmp.nr() / 3, 3);
			// mat_write("test.txt", tmp, "after points");

			// model.print_R();
			// model.print_T();
			// model.print_extrinsic_params();
			// std::cin.get();								
		}
		model.generate_external_parameter();
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


