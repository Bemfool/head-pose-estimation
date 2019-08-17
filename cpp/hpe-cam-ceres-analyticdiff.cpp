/**********************************************************
 * Author: Keith Lin
 * Date: 2019-9-25
 * Description: This demo is used to fitting landmarks got from dlib with a standard
 * model, using Ceres as mathematic tool to deal with the unlinear optimisation. 
 * Since dlib's detector takes rather long time, users should cmake with AVX and SSE2
 * or SSE4, just like:
 * 		cmake .. -D USE_AVX_INSTRUCTIONS=ON
 * 		cmake .. -D USE_SSE2_INSTRUCTIONS=ON
 * 		cmake .. -D USE_SSE4_INSTRUCTIONS=ON
 * And do not forget to build Release version (NO DEBUG), or your video capturing
 * would become stuttering.
 **********************************************************/

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
#include <vector>

using ceres::SizedCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;
using namespace dlib;
using namespace std;

/* Intrinsic parameters (from calibration) */
#define FX 1744.327628674942
#define FY 1747.838275588676
#define CX 800
#define CY 600

/* Number of landmarks */
#define LANDMARK_NUM 68

/* File name of 3d standard model landmarks */
#define LANDMARK_FILE_NAME "landmarks.txt"

/* Definition of 3d coordinate, because dlib only support dlib::point, which is 2d */
typedef dlib::vector<double, 3> point3f;


/* Function: get_landmark
 * Usage: get_landmark(model_landmarks);
 * Parameters: 
 * 		model_landmarks: 3d landmarks coordinates to be saved
 * Returns: true if successful otherwise false
 * --------------------------------------------------------------------------------------------
 * Get 3d coordinates of standard face model's landmarks saved in file named 'landmarks.txt',
 * and save them in model_landmarks.
 */
bool get_landmark(std::vector<point3f>& model_landmarks);


/* Function: landmarks_3d_to_2d
 * Usage: landmarks_3d_to_2d(landmarks_3d, landmarks_2d);
 * Parameters: 
 * 		landmarks_3d: 3d landmarks coordinates
 * 		landmarks_2d: transformed 2d landmarks coordinates to be saved
 * --------------------------------------------------------------------------------------------
 * Transform 3d landmarks into 2d landmarks.
 */
void landmarks_3d_to_2d(std::vector<point3f>& landmarks_3d, std::vector<dlib::point>& landmarks_2d);


/* Function: transform
 * Usage: transform(x);
 * Parameters:
 * 		points: 3d coordinates to be transform
 * 		x: a double array of length 6.
 * 			x[0]: yaw
 * 			x[1]: pitch
 * 			x[2]: roll
 * 			x[3]: tx
 * 			x[4]: ty
 * 			x[5]: tz
 * --------------------------------------------------------------------------------------------
 * Transform a series 3d points to rotate with R(yaw, pitch, roll) and translate with T(tx, ty, tz).
 * Actually, this function encapsulates rotate() and translate() two functions.
 */
void transform(std::vector<point3f>& points, const double * const x);


/* Function: rotate
 * Usage: rotate(points, yaw, pitch, roll);
 * Parameters:
 * 		points: 3d coordinates to be transform
 * 		yaw: angle to rotate with y axis
 * 		pitch: angle to rotate with x axis
 * 		roll: angle to rotate with z axis 
 * --------------------------------------------------------------------------------------------
 * Transform a series 3d points to rotate with R(yaw, pitch, roll)
 */
void rotate(std::vector<point3f>& points, const double yaw, const double pitch, const double roll);


/* Function: translate
 * Usage: translate(points, x, y, z);
 * Parameters:
 * 		points: 3d coordinates to be transform
 * 		x distance to translate along x axis
 * 		y: distance to translate along y axis
 * 		z: distance to translate along z axis
 * --------------------------------------------------------------------------------------------
 * Transform a series 3d points to translate with T(x, y, z)
 */
void translate(std::vector<point3f>& points, const double x, const double y, const double z);


/* Function: split_string
 * Usage: split_string(source, object, separater);
 * Parameters:
 * 		s: source string to be split
 * 		v: a series of string after split
 * 		c: separater
 * --------------------------------------------------------------------------------------------
 * Seperate s with c, and save split series into v.
 */
void split_string(const std::string& s, std::vector<std::string>& v, const std::string& c);

void coarse_estimation(double *x, full_object_detection shape);

/* 3d landmarks coordinates of standard model */
std::vector<point3f> model_landmarks(LANDMARK_NUM);
/* Temp 3d landmarks used for iteration */
std::vector<point3f> fitting_landmarks(LANDMARK_NUM);

/* Cost functor used for Ceres optimisation */
class CostFunctor : public SizedCostFunction<LANDMARK_NUM, 6> {
public:
	CostFunctor(full_object_detection _shape){ shape = _shape; }
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

		/* We use Z1Y2X3 format of Tait–Bryan angles */
		double c1 = cos(roll  * pi / 180.0), s1 = sin(roll  * pi / 180.0);
		double c2 = cos(yaw   * pi / 180.0), s2 = sin(yaw   * pi / 180.0);
		double c3 = cos(pitch * pi / 180.0), s3 = sin(pitch * pi / 180.0);
		// double c1 = cos(roll),  s1 = sin(roll);
		// double c2 = cos(yaw),   s2 = sin(yaw);
		// double c3 = cos(pitch), s3 = sin(pitch);

		for(std::vector<point3f>::iterator iter=fitting_landmarks.begin(); iter!=fitting_landmarks.end(); ++iter) {
			double X = iter->x(), Y = iter->y(), Z = iter->z(); 
			iter->x() = (c1 * c2) * X + (c1 * s2 * s3 - c3 * s1) * Y + (s1 * s3 + c1 * c3 * s2 ) * Z + tx;
			iter->y() = (c2 * s1) * X + (c1 * c3 + s1 * s2 * s3) * Y + (-c3 * s1 * s2 - c1 * s3) * Z + ty;
			iter->z() = (-s2    ) * X + (c2 * s3               ) * Y + (c2 * c3                ) * Z + tz; 
		}
		
		landmarks_3d_to_2d(fitting_landmarks, model_landmarks_2d);
		
		for(int i=0; i<LANDMARK_NUM; i++) {
			long tmp1 = shape.part(i).x() - model_landmarks_2d.at(i).x();
			long tmp2 = shape.part(i).y() - model_landmarks_2d.at(i).y();
			residuals[i] = tmp1 * tmp1 + tmp2 * tmp2;
		}

		if (jacobians != NULL && jacobians[0] != NULL) {
			for(int i=0; i<LANDMARK_NUM; i++) {
				double X0 = model_landmarks.at(i).x();
				double Y0 = model_landmarks.at(i).y();
				double Z0 = model_landmarks.at(i).z();
			
				double X    = ( c1 * c2) * X0 + ( c1 * s2 * s3 - c3 * s1) * Y0 + (s1 * s3 + c1 * c3 * s2) * Z0 + tx;
				double dXd1 = (-s1 * c2) * X0 + (-s1 * s2 * s3 - c3 * c1) * Y0 + (c1 * s3 - s1 * c3 * s2) * Z0;
				double dXd2 = (-c1 * s2) * X0 + ( c1 * c2 * s3 - 0      ) * Y0 + (0       + c1 * c3 * c2) * Z0;
				double dXd3 = ( 0      ) * X0 + ( c1 * c2 * c3 + s3 * s1) * Y0 + (s1 * c3 - c1 * s2 * s2) * Z0;
				
				double Y    = ( c2 * s1) * X0 + ( c1 * c3 + s1 * s2 * s3) * Y0 + (-c3 * s1 * s2 - c1 * s3) * Z0 + ty;
				double dYd1 = ( c2 * c1) * X0 + (-s1 * c3 + c1 * s2 * s3) * Y0 + (-c3 * c1 * s2 + s1 * s3) * Z0;
				double dYd2 = (-s2 * s1) * X0 + ( 0       + s1 * c2 * s3) * Y0 + (-c3 * s1 * c2 - 0      ) * Z0;
				double dYd3 = ( 0      ) * X0 + (-c1 * s3 + s1 * s2 * c3) * Y0 + ( s3 * s1 * s2 - c1 * c3) * Z0;

				double Z    = (-s2     ) * X0 + ( c2 * s3               ) * Y0 + ( c2 * c3               ) * Z0 + tz; 
				double dZd1 = ( 0      ) * X0 + ( 0                     ) * Y0 + ( 0                     ) * Z0;
				double dZd2 = (-c2     ) * X0 + (-s2 * s3               ) * Y0 + (-s2 * c3               ) * Z0;
				double dZd3 = ( 0      ) * X0 + ( c2 * c3               ) * Y0 + (-c2 * s3               ) * Z0;

				double dXdtx = 1, dXdty = 0, dXdtz = 0;
				double dYdtx = 0, dYdty = 1, dYdtz = 0;
				double dZdtx = 0, dZdty = 0, dZdtz = 1;
				
				double dud1  = FX * (dXd1 * Z - X * dZd1) / (Z * Z);
				double dud2  = FX * (dXd2 * Z - X * dZd2) / (Z * Z);
				double dud3  = FX * (dXd3 * Z - X * dZd3) / (Z * Z);
				double dudtx = FX * dXdtx / Z;
				double dudty = 0;
				double dudtz = -FX * X * dZdtz / (Z * Z);

				double dvd1  = FY * (dYd1 * Z - Y * dZd1) / (Z * Z);
				double dvd2  = FY * (dYd2 * Z - Y * dZd2) / (Z * Z);
				double dvd3  = FY * (dYd3 * Z - Y * dZd3) / (Z * Z);
				double dvdtx = 0;
				double dvdty = FY * dYdty / Z;
				double dvdtz = -FY * Y * dZdtz / (Z * Z);

				long residual1 = model_landmarks_2d.at(i).x() - shape.part(i).x();
				long residual2 = model_landmarks_2d.at(i).y() - shape.part(i).y();
				// cout << "residual: " << residual1 << " " << residual2 << endl;

				double para1 = 2.0, para2 = 2.0;
				jacobians[0][i * 6 + 0] = para1 * residual1 * dud2 + para1 * residual2 * dvd2;
				jacobians[0][i * 6 + 1] = para1 * residual1 * dud3 + para1 * residual2 * dvd3;
				jacobians[0][i * 6 + 2] = para1 * residual1 * dud1 + para1 * residual2 * dvd1;

				jacobians[0][i * 6 + 3] = para2 * residual1 * dudtx;
				jacobians[0][i * 6 + 4] = para2 * residual2 * dvdty;
				jacobians[0][i * 6 + 5] = para2 * residual1 * dudtz + para2 * residual2 * dvdtz;
			}
		}

		return true;
	}
private:
	full_object_detection shape;	/* 3d landmarks coordinates got from dlib */
};


int main(int argc, char** argv)
{  
	image_window win;
	try {
		// Init Detector
		cout << "initing detector..." << endl;
		frontal_face_detector detector = get_frontal_face_detector();
		shape_predictor sp;
		deserialize("shape_predictor_68_face_landmarks.dat") >> sp;
		cout << "detector init successfully\n" << endl;

		// Read 3d landmarks coordinates of standard model from file
		cout << "loading model landmarks..." << endl;
		if(get_landmark(model_landmarks)) {
			cout << "model landmarks loaded successfully\n" << endl;
		} else {
			cout << "model landmarks loaded failed\n" << endl;
			return 0;
		}

		/* Use Camera */
		cv::VideoCapture cap(0);
        if (!cap.isOpened())
        {
            cerr << "Unable to connect to camera" << endl;
            return 1;
        }		

		/* Init variables */
		double x[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
		
		while(!win.is_closed()) {
			cv::Mat temp;
			if(!cap.read(temp))
				break;
			dlib::cv_image<bgr_pixel> img(temp);

			/* This type image could not use pyramid_up()， and also I think for talking-head 
			 * video, this is unneccessary.
			 */
			// pyramid_up(img);

			/* Use dlib to detect faces */
			std::vector<rectangle> dets = detector(img);
			cout << "Number of faces detected: " << dets.size() << endl;
	
			std::vector<full_object_detection> shapes;

			for (unsigned long j = 0; j < dets.size(); ++j) {
				/* Use dlib to get landmarks */
				full_object_detection shape = sp(img, dets[j]);

				coarse_estimation(x, shape);

				/* Use Ceres to solve problem */
				Problem problem;
				CostFunction* cost_function = new CostFunctor(shape);
				problem.AddResidualBlock(cost_function, NULL, x);
				Solver::Options options;
				/* [TEST] If video is stuttering, we need to make optimisation caurser */
				// options.max_num_iterations = 10;
				options.minimizer_progress_to_stdout = true;
				Solver::Summary summary;
				Solve(options, &problem, &summary);
				std::cout << summary.BriefReport() << endl;
				std::cout << "x : " << x[0] << " " << x[1] << " " << x[2] << " " << x[3] << " " << x[4] << " " << x[5] << endl;
				
				/* Display final result from fitting
				 * 		Green lines: fit landmarks
				 * 		Blue lines: dlib landmarks
				 */
				double yaw = x[0], pitch = x[1], roll = x[2];
				double tx = x[3], ty = x[4], tz = x[5];
				double c1 = cos(roll  * pi / 180.0), s1 = sin(roll  * pi / 180.0);
				double c2 = cos(yaw   * pi / 180.0), s2 = sin(yaw   * pi / 180.0);
				double c3 = cos(pitch * pi / 180.0), s3 = sin(pitch * pi / 180.0);
				fitting_landmarks.clear();
				for(std::vector<point3f>::iterator iter=fitting_landmarks.begin(); iter!=fitting_landmarks.end(); ++iter) {
					double X = iter->x(), Y = iter->y(), Z = iter->z(); 
					iter->x() = (c1 * c2) * X + (c1 * s2 * s3 - c3 * s1) * Y + (s1 * s3 + c1 * c3 * s2 ) * Z + tx;
					iter->y() = (c2 * s1) * X + (c1 * c3 + s1 * s2 * s3) * Y + (-c3 * s1 * s2 - c1 * s3) * Z + ty;
					iter->z() = (-s2    ) * X + (c2 * s3               ) * Y + (c2 * c3                ) * Z + tz; 
				}
				
				std::vector<point> model_landmarks_2d;
				landmarks_3d_to_2d(fitting_landmarks, model_landmarks_2d);
				std::vector<full_object_detection> model_shapes;
				std::vector<point> parts;
				for(int i=0; i<LANDMARK_NUM; i++)
					parts.push_back(model_landmarks_2d.at(i));
				model_shapes.push_back(full_object_detection(rectangle(), parts));
				win.clear_overlay();
				win.add_overlay(render_face_detections(model_shapes));	
				shapes.push_back(shape);
			}
			/* To avoid the situation that landmarks gets stuck with no face detected */
			if(dets.size()==0)
				win.clear_overlay();

			win.set_image(img);
			win.add_overlay(render_face_detections(shapes, rgb_pixel(0,0, 255)));
	
		}
		// Press any key to exit
		cin.get();
	} catch (exception& e) {
		cout << "\nexception thrown!" << endl;
		cout << e.what() << endl;
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

void coarse_estimation(double *x, full_object_detection shape)
{
	for(int i=0; i<3; i++)
		if(x[i] < 0 || x[i]>360)
			x[i] = 0;
	for(int i=3; i<6; i++)
		if(x[i] < -1e7 || x[i] > 1e7) 
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
	
	for(auto iter=model_landmarks.begin(); iter!=model_landmarks.end(); ++iter) {
		// cout << "before: " << iter->x();
		(*iter) = (*iter) * scale;
		// cout << "after: " << iter->x() << endl;
	}

	/* Estimate yaw angle using line crossing mouth */
	vec = shape.part(49) - shape.part(55);
	model_vec = model_landmarks.at(49) - model_landmarks.at(55); 
	cout << "mouth: " << vec.length() << " " << model_vec.length() << endl;
	cout << vec.length() / model_vec.length() << endl;
	cout << acos(vec.length() / model_vec.length()) << endl; 
	x[0] = acos(vec.length() / model_vec.length()) * 180.0 / pi;
	cout << "yaw: " << x[0] << endl;
}