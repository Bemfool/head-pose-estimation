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

inline dlib::matrix<double> dot(dlib::matrix<double> lhs, dlib::matrix<double> rhs) {
	int length = (lhs.nr()==1) ? lhs.nc() : lhs.nr();
	dlib::matrix<double> res(length, 1);
	for(int i=0; i<length; i++)
		res(i) = lhs(i) * rhs(i);
	return res;
}

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


inline dlib::matrix<double> vec2mat(std::vector<double> &v) {
	dlib::matrix<double> m(v.size(), 1);
	for(int i=0; i<v.size(); i++)
		m(i, 1) = v.at(i);
	return m;
}

inline dlib::matrix<double> vec2mat(std::vector<point3f> &v) {
	dlib::matrix<double> m(v.size() * 3, 1);
	for(int i=0; i<v.size(); i++) {
		m(i * 3, 1)     = v.at(i).x();
		m(i * 3 + 1, 1) = v.at(i).y();
		m(i * 3 + 2, 1) = v.at(i).z();
	}
	return m;
}

inline dlib::matrix<double> vec2mat(std::vector<std::vector<point3f>> &v) {
	int w = v.size(), h = v.at(0).size();
	dlib::matrix<double> m(w, h);
	for(int i=0; i<w; i++) {
		for(int j=0; j<h; j++) {
			m(i, j * 3) = v.at(i).at(j).x();
			m(i, j * 3 + 1) = v.at(i).at(j).y();
			m(i, j * 3 + 2) = v.at(i).at(j).z();
		}
	}
	return m;
}