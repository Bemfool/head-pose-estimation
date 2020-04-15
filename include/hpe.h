#pragma once

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <assert.h>
#include "ceres/ceres.h"
#include "string_utils.h"
#include "io_utils.h"
#include "type_utils.h"
#include "functor/functor.h"


class hpe {
public:
	/* Initialization 
	 * Usage:
	 *     hpe hpe_problem;
	 *     hpe hpe_problem(filename);
	 * Parameters:
	 * 	   @filename: Filename for Basel Face Model loader.
	 *******************************************************************************
	 * Init a head pose estimation problem with input filename as bfm input filename.
	 */
	hpe() {}
	hpe(std::string filename);
	void init(std::string filename);

	bfm &get_model() { return model; }

	/*
	 * Function: solve_ext_params
	 * Usage: hpe_problem.solve_ext_params(FLAG1 | FLAG2 | ...);
	 * Parameters:
	 * 		@mode: Solve mode:
	 * 			- USE_CERES: Using Ceres Solver to solve, using euler angles;
	 * 			- USE_OpenCV: Using OpenCV to solve;
	 * 			- USE_DLT: Using DLT algorithm to estimate intial values. If using
	 * 		      OpenCV or linearized euler angles, it is ON by default.
	 * 			- USE_LINEARIZED_RADIANS: Using linearized euler angle to solve. If using
	 * 			  Ceres Solver, it is OFF by default.
	 * 			This flags can be combined.
	 * 		@ca: Norm coefficient of rotation parameters;
	 * 		@cb: Norm coefficient of translation parameters.
	 *******************************************************************************
	 * Estimate extrinsic parameters using reprojection. And final result will be set
	 * in model.
	 * 
	 */

	bool solve_ext_params(long mode = USE_CERES, double ca = 1.0, double cb = 0.0);
	bool solve_shape_coef();
	bool solve_expr_coef();
	void set_observed_points(dlib::full_object_detection &observed_points_) {observed_points = observed_points_; }
private:
	void dlt();
	bool is_close_enough(double *ext_params, double rotation_eps = 0, double translation_eps = 0);
	dlib::full_object_detection observed_points;
	bfm model;
};