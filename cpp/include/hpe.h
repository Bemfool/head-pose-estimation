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

using ceres::NumericDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;
using namespace dlib;
using namespace std;

/* Definition of 3d coordinate, because dlib only support dlib::point, which is 2d */
typedef dlib::vector<double, 3> point3f;
typedef dlib::point point2d;

/* Intrinsic parameters (from calibration) */
#define FX 1744.327628674942
#define FY 1747.838275588676
#define CX 800
#define CY 600

/* Number of landmarks */
#define LANDMARK_NUM 68

/* File name of 3d standard model landmarks */
#define LANDMARK_FILE_NAME "landmarks.txt"

#define camera_type int
#define PARALLEL    0
#define PINHOLE     1

#define op_type int
#define COARSE  0
#define REAL    1

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

void landmarks_3d_to_2d(camera_type type, std::vector<point3f>& landmarks_3d, std::vector<point2d>& landmarks_2d);

void coarse_estimation(double *x, full_object_detection shape, std::vector<point3f>& _model_landmarks);

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


/* Cost functor used for Ceres optimisation */
struct NumericCostFunctor {
public:
	NumericCostFunctor(full_object_detection _shape, std::vector<point3f> _model_landmarks, op_type _type){ 
        shape = _shape; 
        model_landmarks = _model_landmarks;
        type = _type;
    }
	bool operator()(const double* const x, double* residual) const {
		/* Init landmarks to be transformed */
		std::vector<point3f> fitting_landmarks;
		for(auto iter=model_landmarks.begin(); iter!=model_landmarks.end(); ++iter)
			fitting_landmarks.push_back(*iter);
		std::vector<point> model_landmarks_2d;	
		
		double yaw   = x[0];
		double pitch = x[1];
		double roll  = x[2];
		double tx    = x[3];
		double ty    = x[4];
		double tz    = x[5];

		/* We use Z1Y2X3 format of Tait–Bryan angles */
		double c1 = cos(roll  * pi / 180.0), s1 = sin(roll  * pi / 180.0);
		double c2 = cos(yaw   * pi / 180.0), s2 = sin(yaw   * pi / 180.0);
		double c3 = cos(pitch * pi / 180.0), s3 = sin(pitch * pi / 180.0);
		// double c1 = cos(roll),  s1 = sin(roll);
		// double c2 = cos(yaw),   s2 = sin(yaw);
		// double c3 = cos(pitch), s3 = sin(pitch);

        for(std::vector<point3f>::iterator iter=fitting_landmarks.begin(); iter!=fitting_landmarks.end(); ++iter) {
            double X = iter->x(), Y = iter->y(), Z = iter->z(); 
            iter->x() = ( c1 * c2) * X + (c1 * s2 * s3 - c3 * s1) * Y + ( s1 * s3 + c1 * c3 * s2) * Z + tx;
            iter->y() = ( c2 * s1) * X + (c1 * c3 + s1 * s2 * s3) * Y + (-c3 * s1 * s2 - c1 * s3) * Z + ty;
            iter->z() = (-s2     ) * X + (c2 * s3               ) * Y + ( c2 * c3               ) * Z + tz; 
        }
		
        if(type == REAL)
    		landmarks_3d_to_2d(PINHOLE, fitting_landmarks, model_landmarks_2d);
        else if(type == COARSE)
            landmarks_3d_to_2d(PARALLEL, fitting_landmarks, model_landmarks_2d);
        else {
            cout << "[ERROR] Cost functor init failed." << endl;
            return false;
        }

		/* Calculate the energe (Euclid distance from two points) */
		for(int i=0; i<LANDMARK_NUM; i++) {
			long tmp1 = shape.part(i).x() - model_landmarks_2d.at(i).x();
			long tmp2 = shape.part(i).y() - model_landmarks_2d.at(i).y();
			residual[i] = sqrt(tmp1 * tmp1 + tmp2 * tmp2);
		}
		return true;
	}
private:
	full_object_detection shape;	/* 3d landmarks coordinates got from dlib */
    std::vector<point3f> model_landmarks;
    op_type type;
};

