#include "hpe.h"

void coarse_estimation(double *x, full_object_detection shape, std::vector<point3f>& _model_landmarks)
{
	std::vector<point2d> model_landmarks(LANDMARK_NUM);
	for(int i=0; i<LANDMARK_NUM; i++) {
		model_landmarks.at(i) = point2d(_model_landmarks.at(i).x(), _model_landmarks.at(i).y());
	}

	for(int i=0; i<3; i++)
		if(x[i] < 0 || x[i]>360)
			x[i] = 0;
	for(int i=3; i<6; i++)
		if(x[i] < -1e6 || x[i] > 1e6) 
			x[i] = 0;

	dlib::vector<double, 2> vec;
	dlib::vector<double, 2> model_vec;
	dlib::vector<double, 2> x_axis;
	x_axis.x() = 1;
	x_axis.y() = 0;

	/* Estimate roll angle using line crossing two eyes */
	vec = shape.part(46) - shape.part(37);
	x[2] = acos(vec.dot(x_axis) / (vec.length() * x_axis.length()) ) * 180.0 / pi;
	cout << "roll: " << x[2] << endl;

	/* Estimate scaling size using line crossing two temples */
	vec = shape.part(1) - shape.part(17);
	model_vec = model_landmarks.at(1) - model_landmarks.at(17); 
	double scale = vec.length() / model_vec.length();
	cout << vec.length() << " " << model_vec.length() << endl;
	cout << "scale: " <<  scale << endl;
	
	for(int i=0; i<LANDMARK_NUM; i++) {
		// cout << "before: " << iter->x();
		model_landmarks.at(i) = model_landmarks.at(i) * scale;
		_model_landmarks.at(i) = _model_landmarks.at(i) * scale;
		// cout << "after: " << iter->x() << endl;
	}

	/* Estimate yaw angle using horizonal line crossing mouth */
	vec = shape.part(49) - shape.part(55);
	model_vec = model_landmarks.at(49) - model_landmarks.at(55); 
	cout << "mouth: " << vec.length() << " " << model_vec.length() << endl;
	cout << vec.length() / model_vec.length() << endl;
	cout << acos(vec.length() / model_vec.length()) << endl; 
	if(vec.length() > model_vec.length())
		x[0] = 0.f;
	else {
		dlib::vector<double, 2> vec_left, vec_right;
		vec_left = shape.part(34) - shape.part(32);
		vec_right = shape.part(36) - shape.part(34);
		if(vec_left.length() < vec_right.length()) {
			x[0] = acos(vec.length() / model_vec.length()) * 180.0 / pi;
		} else {
			x[0] = - acos(vec.length() / model_vec.length()) * 180.0 / pi;
		}
	}
	cout << "yaw: " << x[0] << endl;

	/* Esitimate pitch angle using vertical line crossing mouth */
	// vec = shape.part(52) - shape.part(58);
	// model_vec = model_landmarks.at(52) - model_landmarks.at(58);
	// if(vec.length() > model_vec.length())
	// 	x[1] = 0.f;
	// else 
	// 	if(shape.part(34).y() < shape.part(31).y())
	// 		x[1] = acos(vec.length() / model_vec.length()) * 180.0 / pi;	
	// 	else
	// 		x[1] = - acos(vec.length() / model_vec.length()) * 180.0 / pi;	

	/* Estimate tx, ty using nose center */
	x[3] = shape.part(31).x() - model_landmarks.at(31).x();
	x[4] = shape.part(31).y() - model_landmarks.at(31).y();

}

void landmarks_3d_to_2d(camera_type type, std::vector<point3f>& landmarks_3d, std::vector<dlib::point>& landmarks_2d) 
{
	landmarks_2d.clear();
	double xs, ys, zs;
	double xo, yo;
	for(std::vector<point3f>::iterator iter=landmarks_3d.begin(); iter!=landmarks_3d.end(); ++iter) {
		xs = (*iter).x();
		ys = (*iter).y();
		zs = (*iter).z();
		if(type == PINHOLE) {
			xo = FX * xs / zs + CX;
			yo = FY * ys / zs + CY;
		} else if(type == PARALLEL) {
			xo = xs;
			yo = ys;
		} else {
			cout << "[ERROR] Projection from 3d to 2d failed." << endl;
		}
		landmarks_2d.push_back(dlib::point(xo, yo));
	}
}

bool get_landmark(std::vector<point3f>& model_landmarks) 
{
	ifstream in(LANDMARK_FILE_NAME);
	if(in) {
		int count = 0;
		string line;
		while(getline(in, line)) {
			std::vector<string> tmp;
			split_string(line, tmp, " ");
			model_landmarks[count++] = point3f(atof(tmp.at(0).c_str()),
											  -atof(tmp.at(1).c_str()), 
											   atof(tmp.at(2).c_str()));
		}
		return true;
	} else {
		cout << "no LANDMARK_FILE_NAME file" << endl;
		return false;
	}
}

void split_string(const std::string& s, std::vector<std::string>& v, const std::string& c)
{
	std::string::size_type pos1, pos2;
	pos2 = s.find(c);
	pos1 = 0;
	while(std::string::npos != pos2) {
		v.push_back(s.substr(pos1, pos2-pos1));
		pos1 = pos2 + c.size();
		pos2 = s.find(c, pos1);
	}
	if(pos1 != s.length())
		v.push_back(s.substr(pos1));
}


void transform(std::vector<point3f>& points, const double * const x)
{
	rotate(points, x[0], x[1] ,x[2]);
	translate(points, x[3], x[4], x[5]);
}


void rotate(std::vector<point3f>& points, const double yaw, const double pitch, const double roll) 
{
	dlib::point_transform_affine3d around_z = rotate_around_z(roll * pi / 180);
	dlib::point_transform_affine3d around_y = rotate_around_y(yaw * pi / 180);
	dlib::point_transform_affine3d around_x = rotate_around_x(pitch * pi / 180);
	for(std::vector<point3f>::iterator iter=points.begin(); iter!=points.end(); ++iter)
		*iter = around_z(around_y(around_x(*iter)));
}


void translate(std::vector<point3f>& points, const double x, const double y, const double z)
{
	for(std::vector<point3f>::iterator iter=points.begin(); iter!=points.end(); ++iter)
		*iter = (*iter) + point3f(x, y, z);
}
