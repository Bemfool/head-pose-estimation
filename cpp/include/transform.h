#pragma once
#include <cmath>
#include <dlib/matrix.h>
#include <dlib/opencv/to_open_cv.h>
#include <opencv2/opencv.hpp> 
#define _USE_MATH_DEFINES

dlib::matrix<double> matrix_transform(dlib::matrix<double> R, double tx, double ty, double tz, 
					      const dlib::matrix<double> &points);


template<typename T, typename E>
dlib::matrix<T> transform(const T * const ext_parm, const dlib::matrix<E> &before_points) {
    dlib::matrix<T> after_points;
    after_points.set_size(before_points.nr(), 1);

	const T &yaw   = ext_parm[0];
	const T &pitch = ext_parm[1];
	const T &roll  = ext_parm[2];
	const T &tx    = ext_parm[3];
	const T &ty    = ext_parm[4];
	const T &tz    = ext_parm[5];

	/* yaw - phi */
	T c1 = cos(yaw   * T(M_PI) / T(180.0)), s1 = sin(yaw   * T(M_PI) / T(180.0));
	/* pitch - theta */
	T c2 = cos(pitch * T(M_PI) / T(180.0)), s2 = sin(pitch * T(M_PI) / T(180.0));
	/* roll - psi */
	T c3 = cos(roll  * T(M_PI) / T(180.0)), s3 = sin(roll  * T(M_PI) / T(180.0));

	for(int i=0; i<before_points.nr() / 3; i++) {
		T X = T(before_points(i*3)), Y = T(before_points(i*3+1)), Z = T(before_points(i*3+2)); 

		after_points(i*3)   = ( c2 * c1) * X + (s3 * s2 * c1 - c3 * s1) * Y + (c3 * s2 * c1 + s3 * s1) * Z + tx;
		after_points(i*3+1) = ( c2 * s1) * X + (s3 * s2 * s1 + c3 * c1) * Y + (c3 * s2 * s1 - s3 * c1) * Z + ty;
		after_points(i*3+2) = (-s2     ) * X + (s3 * c2               ) * Y + (c3 * c2               ) * Z + tz; 
	}

    return after_points;
}


template<typename T, typename E>
dlib::matrix<T> linearized_transform(const T * const ext_parm, const dlib::matrix<E> &points) {
	dlib::matrix<T> before_points = dlib::matrix_cast<T>(points);
	dlib::matrix<T> after_points;

	const T &yaw   = ext_parm[0];
	const T &pitch = ext_parm[1];
	const T &roll  = ext_parm[2];
	const T &tx    = ext_parm[3];
	const T &ty    = ext_parm[4];
	const T &tz    = ext_parm[5];

	dlib::matrix<T> R(3, 3);
	R = T(1), -yaw, pitch,
	    yaw, T(1), -roll,
		-pitch, roll, T(1);
	// std::cout << R << std::endl;
	int n_vertice = before_points.nr() / 3;
	before_points = dlib::trans(dlib::reshape(before_points, n_vertice, 3));
	// std::cout << "trans: " << before_points(0, 1) << " " << before_points(1, 1) << " " << before_points(2, 1) << std::endl; 
	after_points = R * before_points;
	// std::cout << "trans: " << after_points(0, 1) << " " << after_points(1, 1) << " " << after_points(2, 1) << std::endl; 
	dlib::set_rowm(after_points, 0) = dlib::rowm(after_points, 0) + tx;
	dlib::set_rowm(after_points, 1) = dlib::rowm(after_points, 1) + ty;
	dlib::set_rowm(after_points, 2) = dlib::rowm(after_points, 2) + tz;
	// std::cout << "trans: " << after_points(0, 1) << " " << after_points(1, 1) << " " << after_points(2, 1) << std::endl; 
	after_points = dlib::reshape(dlib::trans(after_points), n_vertice * 3, 1);
    return after_points;
} 



template<typename T>
void transform(const double ext_parm[6], T &x, T &y, T &z) {
	const double &yaw   = ext_parm[0];
	const double &pitch = ext_parm[1];
	const double &roll  = ext_parm[2];
	const double &tx    = ext_parm[3];
	const double &ty    = ext_parm[4];
	const double &tz    = ext_parm[5];

	/* yaw - phi */
	double c1 = cos(yaw   * M_PI / 180.0), s1 = sin(yaw   * M_PI / 180.0);
	/* pitch - theta */
	double c2 = cos(pitch * M_PI / 180.0), s2 = sin(pitch * M_PI / 180.0);
	/* roll - psi */
	double c3 = cos(roll  * M_PI / 180.0), s3 = sin(roll  * M_PI / 180.0);

	T X = x, Y = y, Z = z; 

	x = ( c2 * c1) * X + (s3 * s2 * c1 - c3 * s1) * Y + (c3 * s2 * c1 + s3 * s1) * Z + tx;
	y = ( c2 * s1) * X + (s3 * s2 * s1 + c3 * c1) * Y + (c3 * s2 * s1 - s3 * c1) * Z + ty;
	z = (-s2     ) * X + (s3 * c2               ) * Y + (c3 * c2               ) * Z + tz; 
}


bool is_rotation_matrix(const dlib::matrix<double, 3, 3> &R);


void satisfy_rotation_constraint(dlib::matrix<double, 3, 3> &R);


double mat_norm(dlib::matrix<double> &m);