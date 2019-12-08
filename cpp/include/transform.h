#pragma once
#include <cmath>
#include <dlib/matrix.h>
#define _USE_MATH_DEFINES

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

	T c1 = cos(roll  * T(M_PI) / T(180.0)), s1 = sin(roll  * T(M_PI) / T(180.0));
	T c2 = cos(yaw   * T(M_PI) / T(180.0)), s2 = sin(yaw   * T(M_PI) / T(180.0));
	T c3 = cos(pitch * T(M_PI) / T(180.0)), s3 = sin(pitch * T(M_PI) / T(180.0));

	for(int i=0; i<before_points.nr() / 3; i++) {
		T X = T(before_points(i*3)), Y = T(before_points(i*3+1)), Z = T(before_points(i*3+2)); 

		after_points(i*3)   = ( c1 * c2) * X + (c1 * s2 * s3 - c3 * s1) * Y + ( s1 * s3 + c1 * c3 * s2) * Z + tx;
		after_points(i*3+1) = ( c2 * s1) * X + (c1 * c3 + s1 * s2 * s3) * Y + (-c3 * s1 * s2 - c1 * s3) * Z + ty;
		after_points(i*3+2) = (-s2     ) * X + (c2 * s3               ) * Y + ( c2 * c3               ) * Z + tz; 
	}

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

	double c1 = cos(roll  * M_PI / 180.0), s1 = sin(roll  * M_PI / 180.0);
	double c2 = cos(yaw   * M_PI / 180.0), s2 = sin(yaw   * M_PI / 180.0);
	double c3 = cos(pitch * M_PI / 180.0), s3 = sin(pitch * M_PI / 180.0);

    T X = x, Y = y, Z = z;
    x = ( c1 * c2) * X + (c1 * s2 * s3 - c3 * s1) * Y + ( s1 * s3 + c1 * c3 * s2) * Z + tx;
    y = ( c2 * s1) * X + (c1 * c3 + s1 * s2 * s3) * Y + (-c3 * s1 * s2 - c1 * s3) * Z + ty;
    z = (-s2     ) * X + (c2 * s3               ) * Y + ( c2 * c3               ) * Z + tz; 
}