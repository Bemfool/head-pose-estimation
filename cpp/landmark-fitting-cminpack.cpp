#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/opencv.h>
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <assert.h>
#include <minpack.h>

using namespace dlib;
using namespace std;

// intrinsic parameters
#define FX 1744.327628674942
#define FY 1747.838275588676
#define CX 800
#define CY 600

#define real __minpack_real__

typedef dlib::vector<double,3> point3f;

bool get_landmark(std::vector<point3f>& model_landmarks);
void landmarks_3d_to_2d();
void split_string(const std::string& s, std::vector<std::string>& v, const std::string& c);
void fcn(const int *m, const int *n, const double *x, double *fvec, int *iflag);
void rotate(std::vector<point3f> &points, double yaw, double pitch, double roll);
void transform(std::vector<point3f> &points, double x, double y, double z);

full_object_detection current_shape;
std::vector<point3f> model_landmarks(68);	// 标准模型的特征点三维坐标
std::vector<point3f> fitting_landmarks(68); // 经过旋转、平移的特征点三维坐标
array2d<rgb_pixel> img;

int main(int argc, char** argv)
{  
	image_window win;
	win.clear_overlay();
	try {
		cout << "initing detector..." << endl;
		frontal_face_detector detector = get_frontal_face_detector();
		shape_predictor sp;
		deserialize("shape_predictor_68_face_landmarks.dat") >> sp;
		cout << "detector init successfully" << endl;

		// 读取标准模型的特征点三维坐标
		cout << "loading model landmarks..." << endl;
		if(get_landmark(model_landmarks)) {
			cout << "model landmarks loaded successfully" << endl;
		} else {
			cout << "model landmarks loaded failed" << endl;
			return 0;
		}

		// 将三维坐标根据相机内参转为2维坐标
		// std::vector<point> model_landmarks_2d;
		// landmarks_3d_to_2d(model_landmarks, model_landmarks_2d);		  

		// 用于显示


		// 测试旋转矩阵
		// std::vector<point3f> test_landmarks;
		// for(int i=0; i<68; i++) {
		// 	test_landmarks.push_back(point3f({1.0, 1.0, 0.0}));
		// }
		// rotate(test_landmarks, 45, 0, 0);
		// cout << test_landmarks[0].x() << " " << test_landmarks[0].y() << " " << test_landmarks[0].z() << endl;
		// return 0;

		// 测试位移
		// std::vector<point3f> test_landmarks;
		// for(int i=0; i<68; i++) {
		// 	test_landmarks.push_back(point3f({1.0, 1.0, 0.0}));
		// }
		// transform(test_landmarks, -1, -20, -2);
		// cout << test_landmarks[0].x() << " " << test_landmarks[0].y() << " " << test_landmarks[0].z() << endl;
		// return 0;

		cout << "processing image " << "2019-07-21-123713.jpg" << endl;
		// array2d<rgb_pixel> img;
		load_image(img, "2019-07-21-123713.jpg");
		pyramid_up(img);
		std::vector<rectangle> dets = detector(img);
		cout << "Number of faces detected: " << dets.size() << endl;
		std::vector<full_object_detection> shapes;
		int m = 68, n = 6, info, one = 1;
		int iwa[n], lwa = m * n + 5 * n + m;
		double x[n] = {0, 0, 0, 0, 0, -1000000}, wa[lwa], tol, fvec[m], fnorm;


		for (unsigned long j = 0; j < dets.size(); ++j) {
			full_object_detection shape = sp(img, dets[j]);
			current_shape = shape;
			// tol = sqrt(__minpack_func__(dpmpar)(&one));
			tol = 10000;
			cout << "tol: " << tol << endl;
			__minpack_func__(lmdif1)(&fcn, &m, &n, x, fvec, &tol, &info, iwa, wa, &lwa);
			fnorm = __minpack_func__(enorm)(&m, fvec);
			cout << x[0] << " " << x[1] << " " <<  x[2] <<  " " << x[3] << " " <<  x[4] << " " << x[5]  << endl;
			cout << "final l2 norm of the residuals " << (double)fnorm << endl;
			cout << "exit parameter" << info << endl;
			shapes.push_back(shape);
		}
		
		win.set_title("face poses");
		win.set_image(img);

		win.add_overlay(render_face_detections(shapes, rgb_pixel(0,0, 255)));

		cin.get();
	} catch (exception& e) {
		cout << "\nexception thrown!" << endl;
		cout << e.what() << endl;
	}
}

bool get_landmark(std::vector<point3f>& model_landmarks) 
{
	ifstream in("landmarks.txt");
	string line;
	if(in) {
		int count = 0;
		while(getline(in, line)) {
			// cout << "current input line: " << line << endl;
			std::vector<string> tmp;
			split_string(line, tmp, " ");
			// cout << "after division: " << tmp[0] << " " << tmp[1] << " " << tmp[2] << endl;
			model_landmarks[count].x() = atof(tmp.at(0).c_str());
			model_landmarks[count].y() = atof(tmp.at(1).c_str());	// 负号让人脸变正
			model_landmarks[count].z() = atof(tmp.at(2).c_str());
			count++;
		}
		cout << "0:  " << model_landmarks.at(0).x() << endl;
		cout << "67: " << model_landmarks.at(67).y() << endl;
		return true;
	} else {
		cout << "no landmarks.txt file" << endl;
		return false;
	}
}

void landmarks_3d_to_2d(std::vector<point3f>& landmarks_3d, std::vector<dlib::point>& landmarks_2d) 
{
	double xo, yo, xs, ys, z;
	for(int i=0; i<68; i++) {
		xo = landmarks_3d.at(i).x();
		yo = landmarks_3d.at(i).y();
		z = landmarks_3d.at(i).z();
		xs = FX * xo / z + CX;
		ys = FY * yo / z + CY;
		landmarks_2d.push_back(point(xs, ys));
	}
}


void split_string(const std::string& s, std::vector<std::string>& v, const std::string& c)
{
	// cout << "in " << s << " find " << c << endl;
	std::string::size_type pos1, pos2;
	pos2 = s.find(c);
	// cout << "first pos2: " << pos2 << endl;
	pos1 = 0;
	while(std::string::npos != pos2) {
		v.push_back(s.substr(pos1, pos2-pos1));
		pos1 = pos2 + c.size();
		pos2 = s.find(c, pos1);
	}
	if(pos1 != s.length())
		v.push_back(s.substr(pos1));
	// cout << v[0] << v[1] << v[2] << endl;
}

void fcn(const int *m, const int *n, const double *x, double *fvec, int *iflag)
{
	int i;
	assert(*m == 68 && *n == 6);

	if (*iflag == 0) {
		return;
	}
	
	fitting_landmarks.clear();
	fitting_landmarks.assign(model_landmarks.begin(), model_landmarks.end());
	rotate(fitting_landmarks, x[0], x[1], x[2]);
	transform(fitting_landmarks, x[3], x[4], x[5]);
	std::vector<point> model_landmarks_2d;
	landmarks_3d_to_2d(fitting_landmarks, model_landmarks_2d);
	
	std::vector<full_object_detection> model_shapes;
	std::vector<point> parts;
	for(int i=0; i<68; i++)
		parts.push_back(model_landmarks_2d.at(i));
	model_shapes.push_back(full_object_detection(rectangle(), parts));
	image_window win;
	win.clear_overlay();
	win.set_image(img);
	win.add_overlay(render_face_detections(model_shapes));
	cin.get();

	cout << "fvec: " << endl;
	for (i=0; i<68; i++) {
		cout << "cx: " << current_shape.part(i).x() << " cy: " << current_shape.part(i).y() << " ";
		cout << "mx: " <<model_landmarks_2d.at(i).x() << " my: " << model_landmarks_2d.at(i).y() << endl;
		double tmp1 = current_shape.part(i).x() - model_landmarks_2d.at(i).x();
		double tmp2 = current_shape.part(i).y() - model_landmarks_2d.at(i).y();
		fvec[i] = sqrt(tmp1 * tmp1 + tmp2 * tmp2);
		cout << "fvec[" << i << "] " << fvec[i] << endl;
	}

	cout << "x: " << x[0] << " " << x[1] << " " <<  x[2] <<  " " << x[3] << " " <<  x[4] << " " << x[5]  << endl;

}

void rotate(std::vector<point3f>& points, double yaw, double pitch, double roll) 
{
	// point3f p = points[0];
	// cout << "raw point: " << p.x() <<  " " << p.y() <<  " " << p.z() << endl;
	// p = rotate_around_x(pitch)(p);
	// cout << "after around x: " << pitch << " " << p.x() <<  " " << p.y() <<  " " << p.z() << endl;
	// p = rotate_around_y(yaw)(p);
	// cout << cos(pi/4) << " " << sin(pi/4) << endl;
	// cout << "after around y: " << yaw << " " << p.x() <<  " " << p.y() <<  " " << p.z() << endl;
	// p = rotate_around_z(roll)(p);
	// cout << "after around z: " << roll << " " << p.x() <<  " " << p.y() <<  " " << p.z() << endl;
	 
	for(int i=0; i<68; i++) {
		points.at(i) = rotate_around_z(roll*pi/180)(
					   rotate_around_y(yaw*pi/180)(
					   rotate_around_x(pitch*pi/180)(
					   points.at(i))));
	}	
	// double siny, cosy, sinp, cosp, sinr, cosr;
	// siny = sin(yaw);
	// cosy = cos(yaw);
	// sinp = sin(pitch);
	// cosp = cos(pitch);
	// sinr = sin(roll);
	// cosr = cos(roll);
	// double p1, p2, p3, p4, p5, p6, p7, p8, p9;
	// p1 = cosy * cosp;
	// p2 = cosy * sinp * sinr - siny * cosr;
	// p3 = cosy * sinp * cosr + siny * sinr;
	// p4 = siny * cosp;
	// p5 = siny * sinp * sinr + cosy * cosr;
	// p6 = siny * sinp * cosr - cosy * sinr;
	// p7 = -sinp;
	// p8 = cosp * sinr;
	// p9 = cosp * cosr;
	// point3f new_point;
	// for(int i=0; i<68; i++) {
	// 	new_point.x() = points[i].x() * p1 + points[i].y() * p4 + points[i].z() * p7;
	// 	new_point.y() = points[i].x() * p2 + points[i].y() * p5 + points[i].z() * p8;
	// 	new_point.z() = points[i].x() * p3 + points[i].y() * p6 + points[i].z() * p9;
	// 	points[i].x() = new_point.x();
	// 	points[i].y() = new_point.y();
	// 	points[i].z() = new_point.z();
	// }
}

void transform(std::vector<point3f>& points, double x, double y, double z)
{
	for(int i=0; i<68; i++) {
		points[i].x() += x;
		points[i].y() += y;
		points[i].z() += z;
	}
}
