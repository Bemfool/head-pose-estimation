#pragma once
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
#include "functor/functor.h"

class hpe {
public:
	hpe() {}
	hpe(std::string filename);
	void init(std::string filename);
	bfm &get_model() { return model; }
	void solve_total();
	void solve_ext_parm();
	void solve_shape_coef();
	void solve_expr_coef() {}		// TODO

	void set_observed_points(dlib::full_object_detection &observed_points_) {observed_points = observed_points_; }

private:
	dlib::full_object_detection observed_points;
	bfm model;
};


// /* Function: get_landmark
//  * Usage: get_landmark(model_landmarks);
//  * Parameters: 
//  * 		model_landmarks: 3d landmarks coordinates to be saved
//  * Returns: true if successful otherwise false
//  * --------------------------------------------------------------------------------------------
//  * Get 3d coordinates of standard face model's landmarks saved in file named 'landmarks.txt',
//  * and save them in model_landmarks.
//  */

// bool get_landmark(std::vector<point3f>& model_landmarks);


// void coarse_estimation(double *x, dlib::full_object_detection shape, std::vector<point3f>& _model_landmarks);


// /* Cost functor used for Ceres optimisation */

// struct NumericCostFunctor {
// public:
// 	NumericCostFunctor(dlib::full_object_detection _shape, std::vector<point3f> _model_landmarks, op_type _type);
// 	bool operator()(const double* const x, double* residual) const;
// private:
// 	dlib::full_object_detection shape;	/* 3d landmarks coordinates got from dlib */
//     std::vector<point3f> model_landmarks;
//     op_type type;
// };


// double get_landmarks_by_shape(const double* const x, std::vector<point3f> &landmarks);
// double get_landmarks_by_expr(const double* const x, std::vector<point3f> &landmarks);
// void get_landmarks(std::vector<point3f> &landmarks);
// point3f get_landmark(int idx);