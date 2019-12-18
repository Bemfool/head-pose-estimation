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
#include "functor/functor.h"

class hpe {
public:
	hpe() {}
	hpe(std::string filename);
	void init(std::string filename);
	bfm &get_model() { return model; }
	bool solve_total();
	bool solve_parm();
	bool solve_ext_parm();
	bool solve_ext_parm_coarse() { }	// TODO
	bool solve_int_parm();
	bool solve_shape_coef();
	bool solve_expr_coef();
	void estimate_ext_parm();
	void set_observed_points(dlib::full_object_detection &observed_points_) {observed_points = observed_points_; }
private:
	dlib::full_object_detection observed_points;
	bfm model;
};

