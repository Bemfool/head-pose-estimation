#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <assert.h>
#include "ceres/ceres.h"
#include "string_utils.h"
#include "transform.h"
#include "bfm.h"

using ceres::NumericDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;
using namespace dlib;
using namespace std;

extern dlib::matrix<double> shape_coef;
extern dlib::matrix<double> shape_mu;
extern dlib::matrix<double> shape_ev;
extern dlib::matrix<double> shape_pc;

extern std::vector<point3d> tl;	/* triangle list */

extern dlib::matrix<double> current_shape;

extern std::vector<double> landmark_idx;

/* head pose parameters */
extern double yaw, pitch, roll;	/* rotation */
extern double tx, ty, tz;			/* translation */

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


void coarse_estimation(double *x, full_object_detection shape, std::vector<point3f>& _model_landmarks);


/* Cost functor used for Ceres optimisation */

struct NumericCostFunctor {
public:
	NumericCostFunctor(dlib::full_object_detection _shape, std::vector<point3f> _model_landmarks, op_type _type);
	bool operator()(const double* const x, double* residual) const;
private:
	dlib::full_object_detection shape;	/* 3d landmarks coordinates got from dlib */
    std::vector<point3f> model_landmarks;
    op_type type;
};

struct HeadPoseNumericCostFunctor {
public:
	HeadPoseNumericCostFunctor(dlib::full_object_detection _shape);
	bool operator()(const double* const x, double* residual) const;
private:
	dlib::full_object_detection shape;	/* 3d landmarks coordinates got from dlib */
};

struct ShapeNumericCostFunctor {
public:
	ShapeNumericCostFunctor(dlib::full_object_detection _shape);
	bool operator()(const double* const x, double* residual) const;
private:
	dlib::full_object_detection shape;	/* 3d landmarks coordinates got from dlib */
};

void get_landmarks(const double* const x, std::vector<point3f> &landmarks);
void get_landmarks(std::vector<point3f> &landmarks);