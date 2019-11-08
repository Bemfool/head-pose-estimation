#pragma once
#include <dlib/image_io.h>
#include <dlib/opencv.h>
#include "constant.h"

/* Definition of 3d coordinate, because dlib only support dlib::point, which is 2d */
typedef dlib::vector<double, 3> point3f;
typedef dlib::vector<int, 3> point3d;
typedef dlib::point point2d;

/* Function: dot
 * Usage: std::vector<T> result = dot(a, b);
 * *****************************************************************************
 * Two vectors multiply each other element by element.
 */

template<typename T> std::vector<T> dot(std::vector<T> a, std::vector<T> b) {
	if (a.size() != b.size()) {
		std::cout << "[ERROR] dot size is not compartible." << std::endl;
		std::cout << a.size() << " : " << b.size() << std::endl;
		return a;
	}
	std::vector<T> res;
	for (int i = 0; i<a.size(); i++)
		res.push_back(a.at(i) * b.at(i));
	return res;
}


/* Function: operator*
 * Usage: std::vector<point3f> tmp = pc * ev;
 * *****************************************************************************
 * Used for the multiplication of pc(N_PC*N_VERTICE*3) and ev(N_PC*1)
 */

inline std::vector<point3f> operator*(std::vector<std::vector<point3f>> &a, std::vector<double> &b) {
	std::vector<point3f> res;
	for (int i = 0; i < N_VERTICE; i++) {
		point3f temp;
		for (int j = 0; j < N_PC; j++)
			temp = temp + a[j][i] * b[j];
		res.push_back(temp);
	}
	return res;
}


/* Function: operator+
 * Usage: std::vector<T> result = a + b;
 * *******************************************************************************
 * Add two vectors.
 */

template<typename T> std::vector<T> operator+(const std::vector<T> &lhs, const std::vector<T> &rhs) {
	std::vector<T> res;
	for (int i = 0; i < lhs.size(); i++) 
		res.push_back(lhs.at(i) + rhs.at(i));
	return res;
}