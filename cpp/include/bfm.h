#pragma once
#include <iostream>
#include <vector>
#include "data.h"
#include "random.h"
#include "vec.h"

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

/* For Users */
bool init_bfm();
void generate_random_face(double scale = 1.0);
void generate_face();
void ply_write(std::string fn);

/* For Programmers */
void load_landmark_idx();
dlib::matrix<double> coef2object(dlib::matrix<double> &coef, dlib::matrix<double> &mu, 
								 dlib::matrix<double> &pc, dlib::matrix<double> &ev);
std::vector<point3f> coef2object(std::vector<double> &coef, 
	std::vector<point3f> &mu, std::vector<std::vector<point3f>> &pc, std::vector<double> &ev);