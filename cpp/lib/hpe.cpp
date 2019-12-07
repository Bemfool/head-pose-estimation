#include "hpe.h"

hpe::hpe(std::string filename) {
	init(filename);
}

void hpe::init(std::string filename) {
	model.init(filename);
}


void hpe::solve_ext_parm() {
	ceres::Problem problem;
	dlib::matrix<double> fp_shape = model.get_fp_current_blendshape();
	ceres::CostFunction *cost_function = ext_parm_reproj_err::create(observed_points, model);
	problem.AddResidualBlock(cost_function, nullptr, model.get_mutable_external_parm());
	ceres::Solver::Options options;
	options.num_threads = 8;
	options.minimizer_progress_to_stdout = true;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.BriefReport() << std::endl;
}






// /************************ Numeric  Cost Functor ******************************/

// NumericCostFunctor::NumericCostFunctor(full_object_detection _shape, 
// 									   std::vector<point3f> _model_landmarks, 
// 									   op_type _type){ 
// 	shape = _shape; 
// 	model_landmarks = _model_landmarks;
// 	type = _type;
// }


// bool NumericCostFunctor::operator()(const double* const x, double* residual) const {
// 	/* Init landmarks to be transformed */
// 	std::vector<point3f> fitting_landmarks;
// 	for(auto iter=model_landmarks.begin(); iter!=model_landmarks.end(); ++iter)
// 		fitting_landmarks.push_back(*iter);
// 	std::vector<point2d> model_landmarks_2d;	
	
// 	double yaw   = x[0];
// 	double pitch = x[1];
// 	double roll  = x[2];
// 	double tx    = x[3];
// 	double ty    = x[4];
// 	double tz    = x[5];

// 	/* We use Z1Y2X3 format of Tait–Bryan angles */
// 	double c1 = cos(roll  * pi / 180.0), s1 = sin(roll  * pi / 180.0);
// 	double c2 = cos(yaw   * pi / 180.0), s2 = sin(yaw   * pi / 180.0);
// 	double c3 = cos(pitch * pi / 180.0), s3 = sin(pitch * pi / 180.0);
// 	// double c1 = cos(roll),  s1 = sin(roll);
// 	// double c2 = cos(yaw),   s2 = sin(yaw);
// 	// double c3 = cos(pitch), s3 = sin(pitch);

// 	for(std::vector<point3f>::iterator iter=fitting_landmarks.begin(); iter!=fitting_landmarks.end(); ++iter) {
// 		double X = iter->x(), Y = iter->y(), Z = iter->z(); 
// 		iter->x() = ( c1 * c2) * X + (c1 * s2 * s3 - c3 * s1) * Y + ( s1 * s3 + c1 * c3 * s2) * Z + tx;
// 		iter->y() = ( c2 * s1) * X + (c1 * c3 + s1 * s2 * s3) * Y + (-c3 * s1 * s2 - c1 * s3) * Z + ty;
// 		iter->z() = (-s2     ) * X + (c2 * s3               ) * Y + ( c2 * c3               ) * Z + tz; 
// 	}
	
// 	if(type == REAL)
// 		landmarks_3d_to_2d(PINHOLE, fitting_landmarks, model_landmarks_2d);
// 	else if(type == COARSE)
// 		landmarks_3d_to_2d(PARALLEL, fitting_landmarks, model_landmarks_2d);
// 	else {
// 		cout << "[ERROR] Cost functor init failed." << endl;
// 		return false;
// 	}

// 	/* Calculate the energe (Euclid distance from two points) */
// 	for(int i=0; i<LANDMARK_NUM; i++) {
// 		long tmp1 = shape.part(i).x() - model_landmarks_2d.at(i).x();
// 		long tmp2 = shape.part(i).y() - model_landmarks_2d.at(i).y();
// 		residual[i] = sqrt(tmp1 * tmp1 + tmp2 * tmp2);
// 	}
// 	return true;
// }


// void coarse_estimation(double *x, full_object_detection shape, std::vector<point3f>& _model_landmarks)
// {
// 	std::vector<point2d> model_landmarks(LANDMARK_NUM);
// 	for(int i=0; i<LANDMARK_NUM; i++) {
// 		model_landmarks.at(i) = point2d(_model_landmarks.at(i).x(), _model_landmarks.at(i).y());
// 	}

// 	for(int i=0; i<3; i++)
// 		if(x[i] < 0 || x[i]>360)
// 			x[i] = 0;
// 	for(int i=3; i<6; i++)
// 		if(x[i] < -1e6 || x[i] > 1e6) 
// 			x[i] = 0;

// 	dlib::vector<double, 2> vec;
// 	dlib::vector<double, 2> model_vec;
// 	dlib::vector<double, 2> x_axis;
// 	x_axis.x() = 1;
// 	x_axis.y() = 0;

// 	/* Estimate roll angle using line crossing two eyes */
// 	vec = shape.part(46) - shape.part(37);
// 	x[2] = acos(vec.dot(x_axis) / (vec.length() * x_axis.length()) ) * 180.0 / pi;
// 	cout << "roll: " << x[2] << endl;

// 	/* Estimate scaling size using line crossing two temples */
// 	vec = shape.part(1) - shape.part(17);
// 	model_vec = model_landmarks.at(1) - model_landmarks.at(17); 
// 	double scale = vec.length() / model_vec.length();
// 	cout << vec.length() << " " << model_vec.length() << endl;
// 	cout << "scale: " <<  scale << endl;
	
// 	for(int i=0; i<LANDMARK_NUM; i++) {
// 		// cout << "before: " << iter->x();
// 		model_landmarks.at(i) = model_landmarks.at(i) * scale;
// 		_model_landmarks.at(i) = _model_landmarks.at(i) * scale;
// 		// cout << "after: " << iter->x() << endl;
// 	}

// 	/* Estimate yaw angle using horizonal line crossing mouth */
// 	vec = shape.part(49) - shape.part(55);
// 	model_vec = model_landmarks.at(49) - model_landmarks.at(55); 
// 	cout << "mouth: " << vec.length() << " " << model_vec.length() << endl;
// 	cout << vec.length() / model_vec.length() << endl;
// 	cout << acos(vec.length() / model_vec.length()) << endl; 
// 	if(vec.length() > model_vec.length())
// 		x[0] = 0.f;
// 	else {
// 		dlib::vector<double, 2> vec_left, vec_right;
// 		vec_left = shape.part(34) - shape.part(32);
// 		vec_right = shape.part(36) - shape.part(34);
// 		if(vec_left.length() < vec_right.length()) {
// 			x[0] = acos(vec.length() / model_vec.length()) * 180.0 / pi;
// 		} else {
// 			x[0] = - acos(vec.length() / model_vec.length()) * 180.0 / pi;
// 		}
// 	}
// 	std::cout << "yaw: " << x[0] << std::endl;

// 	/* Esitimate pitch angle using vertical line crossing mouth */
// 	// vec = shape.part(52) - shape.part(58);
// 	// model_vec = model_landmarks.at(52) - model_landmarks.at(58);
// 	// if(vec.length() > model_vec.length())
// 	// 	x[1] = 0.f;
// 	// else 
// 	// 	if(shape.part(34).y() < shape.part(31).y())
// 	// 		x[1] = acos(vec.length() / model_vec.length()) * 180.0 / pi;	
// 	// 	else
// 	// 		x[1] = - acos(vec.length() / model_vec.length()) * 180.0 / pi;	

// 	/* Estimate tx, ty using nose center */
// 	x[3] = shape.part(31).x() - model_landmarks.at(31).x();
// 	x[4] = shape.part(31).y() - model_landmarks.at(31).y();

// }


// double get_landmarks_by_shape(const double* const x, std::vector<point3f> &landmarks) {
// 	dlib::matrix<double> tmp(N_ID_PC, 1);
// 	for(int i=0; i<N_ID_PC; i++)
// 		tmp(i) = x[i];
// 	current_shape = coef2object(tmp, shape_mu, shape_pc, shape_ev);
// 	current_blendshape = current_shape + current_expr;
// 	get_landmarks(landmarks);
// 	return sum(abs(tmp));
// }

// double get_landmarks_by_expr(const double* const x, std::vector<point3f> &landmarks) {
// 	dlib::matrix<double> tmp(N_EXPR_PC, 1);
// 	for(int i=0; i<N_EXPR_PC; i++)
// 		tmp(i) = x[i];
// 	current_expr = coef2object(tmp, expr_mu, expr_pc, expr_ev);
// 	current_blendshape = current_shape + current_expr;
// 	get_landmarks(landmarks);
// 	return sum(abs(tmp));
// }



// void get_landmarks(std::vector<point3f> &landmarks) {
// 	for(int i=0; i<LANDMARK_NUM; i++) {
// 		landmarks[i].x() = current_blendshape(landmark_idx[i] * 3, 0);
// 		landmarks[i].y() = current_blendshape(landmark_idx[i] * 3 + 1, 0);
// 		landmarks[i].z() = current_blendshape(landmark_idx[i] * 3 + 2, 0);
// 	}
// }

// point3f get_landmark(int idx) {
// 	point3f tmp;
// 	tmp.x() = current_blendshape(landmark_idx[idx] * 3, 0);
// 	tmp.y() = current_blendshape(landmark_idx[idx] * 3 + 1, 0);
// 	tmp.z() = current_blendshape(landmark_idx[idx] * 3 + 2, 0);
// 	return tmp;
// }