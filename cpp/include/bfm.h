#pragma once
#include <iostream>
#include <vector>
#include "data.h"
#include "random.h"
#include "vec.h"

/* data collection */
extern dlib::matrix<double> shape_coef;
extern dlib::matrix<double> shape_mu;
extern dlib::matrix<double> shape_ev;
extern dlib::matrix<double> shape_pc;

extern dlib::matrix<double> tex_coef;
extern dlib::matrix<double> tex_mu;
extern dlib::matrix<double> tex_ev;
extern dlib::matrix<double> tex_pc;

extern dlib::matrix<double> expr_coef;
extern dlib::matrix<double> expr_mu;
extern dlib::matrix<double> expr_ev;
extern dlib::matrix<double> expr_pc;

extern std::vector<point3d> tl;	/* triangle list */

extern dlib::matrix<double> current_shape;
extern dlib::matrix<double> current_tex;
extern dlib::matrix<double> current_expr;
extern dlib::matrix<double> current_blendshape;

extern std::vector<double> landmark_idx;

/* head pose parameters */
extern double yaw, pitch, roll;	/* rotation */
extern double tx, ty, tz;			/* translation */

/* For Users */
bool init_bfm();
void generate_random_face(double scale = 1.0);
void generate_random_face(double shape_scale, double tex_scale, double expr_scale);
void generate_face();
void ply_write(std::string fn);

/* For Programmers */
void load_landmark_idx();
dlib::matrix<double> coef2object(dlib::matrix<double> &coef, dlib::matrix<double> &mu, 
								 dlib::matrix<double> &pc, dlib::matrix<double> &ev);