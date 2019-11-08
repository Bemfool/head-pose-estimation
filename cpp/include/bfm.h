#pragma once
#include <iostream>
#include <vector>
#include "data.h"
#include "random.h"
#include "vec.h"

extern std::vector<double> alpha;
extern std::vector<double> beta;

extern std::vector<point3f> shape_mu;
extern std::vector<double> shape_ev;
extern std::vector<std::vector<point3f>> shape_pc;

extern std::vector<point3f> tex_mu;
extern std::vector<double> tex_ev;
extern std::vector<std::vector<point3f>> tex_pc;

extern std::vector<point3d> tl;

extern std::vector<point3f> shape;
extern std::vector<point3f> tex;

/* For Users */
bool init_bfm();
void generate_random_face(double scale = 1.0);
void generate_face();
void save_ply(string filename);

/* For Programmers */
std::vector<point3f> coef2object(std::vector<double> &coef, 
	std::vector<point3f> &mu, std::vector<std::vector<point3f>> &pc, std::vector<double> &ev);
void ply_write(std::string fn);