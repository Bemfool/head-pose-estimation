#pragma once
#include <cmath>
#include <dlib/matrix.h>
#include <dlib/opencv/to_open_cv.h>
#include <opencv2/opencv.hpp> 
#define _USE_MATH_DEFINES

dlib::matrix<double> matrix_transform(dlib::matrix<double> R, double tx, double ty, double tz, 
					      const dlib::matrix<double> &points);


/* 
 * Function: euler2matrix
 * Usage: dlib::matrix<T> R = euler2matrix(yaw, pitch, roll, false);
 * Parameters:
 * 		@yaw: Euler angle ;
 * 		@pitch: Euler angle;
 * 		@roll: Euler angle;
 * 		@is_linearized: Choose to use linearized Euler angle transform or not. If true, be sure yaw, pitch and roll
 * 						keep small.
 * Return:
 * 		Rotation matrix.
 * 		If linearized:
 * 			R = [[1,      -yaw, pitch],
 * 				 [yaw,    1,    -roll],
 * 				 [-pitch, roll, 1    ]]
 * 		Else
 *			R = [[c2*c1, s3*s2*c1-c3*s1, c3*s2*c1+s3*s1],
 *				 [c2*s1, s3*s2*s1+c3*c1, c3*s2*s1-s3*c1],
 *				 [-s2,   s3*c2,          c3*c2         ]]; 
 *			(c1=cos(yaw), s1=sin(yaw))
 *			(c2=cos(pitch), s2=sin(pitch))
 *			(c3=cos(roll), s3=sin(roll))
 * ----------------------------------------------------------------------------------------------------------------
 * Transform Euler angle into rotation matrix.
 * 
 */

template<typename _Tp> inline 
dlib::matrix<_Tp> euler2matrix(const _Tp &yaw, const _Tp &pitch, const _Tp &roll, bool is_linearized = false)
{
	/* yaw - phi */	
	/* pitch - theta */		
	/* roll - psi */
	dlib::matrix<_Tp> R(3, 3);
	if(is_linearized)
	{
		R = _Tp(1.0), -yaw,     pitch,
	  	    yaw,      _Tp(1.0), -roll,
		    -pitch,   roll,     _Tp(1.0);
	}
	else
	{
		_Tp c1 = cos(yaw   * _Tp(M_PI) / _Tp(180.0)), s1 = sin(yaw   * _Tp(M_PI) / _Tp(180.0));
		_Tp c2 = cos(pitch * _Tp(M_PI) / _Tp(180.0)), s2 = sin(pitch * _Tp(M_PI) / _Tp(180.0));
		_Tp c3 = cos(roll  * _Tp(M_PI) / _Tp(180.0)), s3 = sin(roll  * _Tp(M_PI) / _Tp(180.0));
		R = c2 * c1, s3 * s2 * c1 - c3 * s1, c3 * s2 * c1 + s3 * s1,
			c2 * s1, s3 * s2 * s1 + c3 * c1, c3 * s2 * s1 - s3 * c1,
			-s2,     s3 * c2,                c3 * c2; 
	}
	return R;
}


/* 
 * Function: ternary2quaternion
 * Usage: dlib::matrix<T> quaternion_points = ternary2quaternion(ternary_points);
 * Parameters:
 * 		@ternary_points: Point matrix whose size is (n_vertice, 3).
 * Return:
 * 		Point matrix whose size is (4, n_vertice).
 * 		$Qmat = [Tmat | 1] ^ T$
 * ----------------------------------------------------------------------------------------------------------------
 * Transform a ternary point vector into quaternion vector. Detailed steps are as follows:
 * 1) Add ones matrix in last column;
 * 2) Transpose;
 * 
 */

template<typename _Tp> inline
dlib::matrix<_Tp> ternary2quaternion(const dlib::matrix<_Tp> &ternary_points)
{
	dlib::matrix<_Tp> quaternion_points;
	quaternion_points.set_size(ternary_points.nc()+1, 3);
	dlib::set_subm(quaternion_points, dlib::range(0, ternary_points.nc()-1), dlib::range(0, 2)) = ternary_points;
	dlib::set_colm(quaternion_points, 3) = (_Tp)1.0;	
	return dlib::trans(quaternion_points);
}


/* 
 * Function: RT2P
 * Usage: dlib::matrix<T> P = RT2P(R, T);
 * Parameters:
 * 		@R: Rotation matrix whose size is (3, 3);
 * 		@T: Translation vector whose size is (3, 1);
 * Return:
 * 		Translation affine matrix.
 * 		$P = [R | T]$
 * ----------------------------------------------------------------------------------------------------------------
 * Concate R and T into P.
 * 
 */

template<typename _Tp> inline
dlib::matrix<_Tp> RT2P(const dlib::matrix<_Tp> &R, const dlib::matrix<_Tp> &T)
{
	dlib::matrix<_Tp> P;
	P.set_size(3, 4);
	dlib::set_subm(P, dlib::range(0, 2), dlib::range(0, 2)) = R;
	dlib::set_colm(P, 3) = T;
	return P;
}



template<typename _Tp, typename _Ep>
dlib::matrix<_Tp> transform(const _Tp * const ext_parm, const dlib::matrix<_Ep> &points, bool is_linearized = false) 
{
	dlib::matrix<_Tp> R, T, P;
    dlib::matrix<_Tp> tmp_points, before_points, after_points;

	const _Tp &yaw   = ext_parm[0];
	const _Tp &pitch = ext_parm[1];
	const _Tp &roll  = ext_parm[2];
	const _Tp &tx    = ext_parm[3];
	const _Tp &ty    = ext_parm[4];
	const _Tp &tz    = ext_parm[5];
	
	R = euler2matrix(yaw, pitch, roll, is_linearized);
	T = tx, ty, tz;
	P = RT2P(R, T);

	tmp_points = dlib::reshape(dlib::matrix_cast<_Tp>(points), points.nc()/3, 3);
	before_points = ternary2quaternion(tmp_points);
	after_points = R * P;
    return dlib::reshape(after_points, points.nc(), 1);
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