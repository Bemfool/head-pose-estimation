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
#include "ceres/ceres.h"
#include "glog/logging.h"
#include <vector>

using ceres::SizedCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;
using namespace dlib;
using namespace std;

// intrinsic parameters
#define FX 1744.327628674942
#define FY 1747.838275588676
#define CX 800
#define CY 600

#define LANDMARK_NUM 68

typedef dlib::vector<double, 3> point3f;

bool get_landmark(std::vector<point3f>& model_landmarks);
void landmarks_3d_to_2d(std::vector<point3f>& landmarks_3d, std::vector<dlib::point>& landmarks_2d);
void split_string(const std::string& s, std::vector<std::string>& v, const std::string& c);
void transform(std::vector<point3f>& points, const double * const x);
void rotate(std::vector<point3f>& points, const double yaw, const double pitch, const double roll);
void translate(std::vector<point3f>& points, const double x, const double y, const double z);

full_object_detection current_shape;
std::vector<point3f> model_landmarks(LANDMARK_NUM);	// 标准模型的特征点三维坐标
std::vector<point3f> fitting_landmarks(LANDMARK_NUM); // 经过旋转、平移的特征点三维坐标
array2d<rgb_pixel> img;

class CostFunctor : public SizedCostFunction<LANDMARK_NUM, 6> {
public:
	virtual ~CostFunctor() {}
  	virtual bool Evaluate(double const* const* parameters,
                       	  double* residuals,
                          double** jacobians) const {
		double yaw   = parameters[0][0];
		double pitch = parameters[0][1];
		double roll  = parameters[0][2];
		double tx    = parameters[0][3];
		double ty    = parameters[0][4];
		double tz    = parameters[0][5];

		fitting_landmarks.clear();
		for(std::vector<point3f>::iterator iter=model_landmarks.begin(); iter!=model_landmarks.end(); ++iter)
			fitting_landmarks.push_back(*iter);
		std::vector<point> model_landmarks_2d;

		double cos_yaw   = cos(yaw * pi / 180.0), sin_yaw   = sin(yaw * pi / 180.0);
		double cos_pitch = cos(pitch * pi / 180.0), sin_pitch = sin(pitch * pi / 180.0);
		double cos_roll  = cos(roll * pi / 180.0), sin_roll  = sin(roll * pi / 180.0);

		for(std::vector<point3f>::iterator iter=fitting_landmarks.begin(); iter!=fitting_landmarks.end(); ++iter) {
			double X = iter->x(), Y = iter->y(), Z = iter->z(); 
			iter->x() = cos_roll * cos_yaw * X + cos_roll * sin_yaw * Y - cos_roll * sin_yaw * sin_pitch * Z - sin_roll * cos_pitch * Y + sin_roll * sin_pitch * Z;
			iter->y() = sin_roll * cos_yaw * X + sin_roll * sin_yaw * Y - sin_roll * sin_yaw * sin_pitch * Z + cos_roll * cos_pitch * Y - cos_roll * sin_pitch * Z;
			iter->z() = -sin_yaw * X + cos_yaw * sin_pitch * Y + cos_yaw * cos_pitch * Z; 
		}

		for(std::vector<point3f>::iterator iter=fitting_landmarks.begin(); iter!=fitting_landmarks.end(); ++iter) {
			iter->x() = iter->x() + tx;
			iter->y() = iter->y() + ty;
			iter->z() = iter->z() + tz;
		}
		
		landmarks_3d_to_2d(fitting_landmarks, model_landmarks_2d);
		
		for(int i=0; i<LANDMARK_NUM; i++) {
			long tmp1 = current_shape.part(i).x() - model_landmarks_2d.at(i).x();
			long tmp2 = current_shape.part(i).y() - model_landmarks_2d.at(i).y();
			residuals[i] = tmp1 * tmp1 + tmp2 * tmp2;
		}
		cout << "rediduals caculation done." << endl;

		if (jacobians != NULL && jacobians[0] != NULL) {
			for(int i=0; i<LANDMARK_NUM; i++) {
				double X0 = model_landmarks.at(i).x();
				double Y0 = model_landmarks.at(i).y();
				double Z0 = model_landmarks.at(i).z();
				double X = cos_roll * cos_yaw * X0 + cos_roll * sin_yaw * Y0 - cos_roll * sin_yaw * sin_pitch * Z0 - sin_roll * cos_pitch * Y0 + sin_roll * sin_pitch * Z0
						   + tx;
				double Y = sin_roll * cos_yaw * X0 + sin_roll * sin_yaw * Y0 - sin_roll * sin_yaw * sin_pitch * Z0 + cos_roll * cos_pitch * Y0 - cos_roll * sin_pitch * Z0
				           + ty;
				double Z = -sin_yaw * X0 + cos_yaw * sin_pitch * Y0 + cos_yaw * cos_pitch * Z0
						   + tz; 

				double dXdp = - cos_roll * sin_yaw * sin_pitch * Y0 - cos_roll * sin_yaw * cos_pitch * Z0 + sin_roll * sin_pitch * Y0 + sin_roll * cos_pitch * Z0;
				double dXdy = - cos_roll * sin_yaw * X0 + cos_roll * cos_yaw * cos_pitch * Y0 - cos_roll * cos_yaw * sin_pitch * Z0;
				double dXdr = sin_roll * cos_yaw * X0 - sin_roll * sin_yaw * cos_pitch * Y0 + sin_roll * sin_yaw * sin_pitch * Z0 - cos_roll * cos_pitch * Y0 + cos_roll * sin_pitch * Z0;
				double dXdtx = 1, dXdty = 0, dXdtz = 0;

				double dYdp = - sin_roll * sin_yaw * sin_pitch * Y0 - sin_roll * sin_yaw * cos_pitch * Z0 - cos_roll * sin_pitch * Y0 - cos_roll * cos_pitch * Z0;
				double dYdy = - sin_roll * sin_yaw * X0 + sin_roll * cos_yaw * cos_pitch * Y0 - sin_roll * cos_yaw * sin_pitch * Z0;
				double dYdr = cos_roll * sin_yaw * X0 + cos_roll * sin_yaw * cos_pitch * Y0 - cos_roll * sin_yaw * sin_pitch * Z0 - sin_roll * cos_yaw * Y0 + sin_roll * sin_pitch * Z0;
				double dYdtx = 0, dYdty = 1, dYdtz = 0;

				double dZdp = cos_yaw * cos_pitch * Y0 - cos_yaw * sin_pitch * Z0;
				double dZdy = - cos_yaw * X0 - sin_yaw * sin_pitch * Y0 - sin_yaw * cos_pitch * Z0;
				double dZdr = 0;
				double dZdtx = 0, dZdty = 0, dZdtz = 1;

				double dudp = FX * (dXdp * Z - X * dZdp) / (Z * Z);
				double dudy = FX * (dXdy * Z - X * dZdy) / (Z * Z);
				double dudr = FX * (dXdr * Z - X * dZdr) / (Z * Z);
				double dudtx = FX * dXdtx / Z;
				double dudty = 0;
				double dudtz = -FX * X * dZdtz / (Z * Z);

				double dvdp = FY * (dYdp * Z - Y * dZdp) / (Z * Z);
				double dvdy = FY * (dYdy * Z - Y * dZdy) / (Z * Z);
				double dvdr = FY * (dYdr * Z - Y * dZdr) / (Z * Z);
				double dvdtx = 0;
				double dvdty = FY * dYdty / Z;
				double dvdtz = -FY * Y * dZdtz / (Z * Z);

				long residual1 = model_landmarks_2d.at(i).x() - current_shape.part(i).x();
				long residual2 = model_landmarks_2d.at(i).y() - current_shape.part(i).y();

				double para = 2.0;
				jacobians[0][i * 6 + 0] = para * residual1 * dudy + para * residual2 * dvdy;
				jacobians[0][i * 6 + 1] = para * residual1 * dudp + para * residual2 * dvdp;
				jacobians[0][i * 6 + 2] = para * residual1 * dudr + para * residual2 * dvdr;
				jacobians[0][i * 6 + 3] = para * residual1 * dudtx;
				jacobians[0][i * 6 + 4] = para * residual2 * dvdty;
				jacobians[0][i * 6 + 5] = para * residual1 * dudtz + para * residual2 * dvdtz;
			}
		}

		return true;
	}
};

int main(int argc, char** argv)
{  
	google::InitGoogleLogging(argv[0]);
	image_window win;
	win.clear_overlay();
	try {
		// 初始化detector
		cout << "initing detector..." << endl;
		frontal_face_detector detector = get_frontal_face_detector();
		shape_predictor sp;
		deserialize("shape_predictor_68_face_landmarks.dat") >> sp;
		cout << "detector init successfully\n" << endl;

		// 读取标准模型的特征点三维坐标
		cout << "loading model landmarks..." << endl;
		if(get_landmark(model_landmarks)) {
			cout << "model landmarks loaded successfully\n" << endl;
		} else {
			cout << "model landmarks loaded failed\n" << endl;
			return 0;
		}

		// 打开图片获得人脸框选
		// cout << "processing image " << "2019-07-21-123713.jpg" << endl;
		// load_image(img, "2019-07-21-123713.jpg");
		cout << "processing image " << "2019-07-23-214644.jpg" << endl;
		load_image(img, "2019-07-23-214644.jpg");


		pyramid_up(img);
		std::vector<rectangle> dets = detector(img);
		cout << "Number of faces detected: " << dets.size() << endl;
		std::vector<full_object_detection> shapes;
		win.set_image(img);
		for (unsigned long j = 0; j < dets.size(); ++j) {
			// 将当前获得的特征点数据放置到全局
			full_object_detection shape = sp(img, dets[j]);
			current_shape = shape;

			// 变量定义以及初始化
			double x[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 100000};

			Problem problem;
			CostFunction* cost_function = new CostFunctor;
			problem.AddResidualBlock(cost_function, NULL, x);
			Solver::Options options;
			// options.gradient_tolerance = true;
			options.minimizer_progress_to_stdout = true;
			Solver::Summary summary;
			cout << "??" << endl;
			Solve(options, &problem, &summary);
			std::cout << summary.BriefReport() << endl;
			std::cout << "x : " << x[0] << " " << x[1] << " " << x[2] << " " << x[3] << " " << x[4] << " " << x[5] << endl;

			fitting_landmarks.clear();
			for(std::vector<point3f>::iterator iter=model_landmarks.begin(); iter!=model_landmarks.end(); ++iter)
				fitting_landmarks.push_back(*iter);
			transform(fitting_landmarks, x);
			std::vector<point> model_landmarks_2d;
			landmarks_3d_to_2d(fitting_landmarks, model_landmarks_2d);
			std::vector<full_object_detection> model_shapes;
			std::vector<point> parts;
			for(int i=0; i<LANDMARK_NUM; i++)
				parts.push_back(model_landmarks_2d.at(i));
			model_shapes.push_back(full_object_detection(rectangle(), parts));
			win.add_overlay(render_face_detections(model_shapes));	
			shapes.push_back(shape);
		}
		
		// image_window dlib_win;
		// // 显示dlib获取的特征点
		// dlib_win.set_title("face poses");
		// dlib_win.set_image(img);
		win.add_overlay(render_face_detections(shapes, rgb_pixel(0,0, 255)));

		// 任意键退出
		cin.get();
	} catch (exception& e) {
		cout << "\nexception thrown!" << endl;
		cout << e.what() << endl;
	}
}

/* 
 * 用处: 从"landmarks.txt"文件中读取标准模型的landmark三维坐标
 */

bool get_landmark(std::vector<point3f>& model_landmarks) 
{
	ifstream in("landmarks.txt");
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
		cout << "67: " << model_landmarks[67].x() << endl;
		return true;
	} else {
		cout << "no landmarks.txt file" << endl;
		return false;
	}
}

void landmarks_3d_to_2d(std::vector<point3f>& landmarks_3d, std::vector<dlib::point>& landmarks_2d) 
{
	landmarks_2d.clear();
	double xs, ys, zs;
	double xo, yo;
	for(std::vector<point3f>::iterator iter=landmarks_3d.begin(); iter!=landmarks_3d.end(); ++iter) {
		xs = (*iter).x();
		ys = (*iter).y();
		zs = (*iter).z();
		xo = FX * xs / zs + CX;
		yo = FY * ys / zs + CY;
		landmarks_2d.push_back(dlib::point(xo, yo));
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
