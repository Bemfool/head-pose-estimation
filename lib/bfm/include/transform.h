#pragma once
#define _USE_MATH_DEFINES
#include <cmath>
#include <dlib/matrix.h>
#include <dlib/opencv/to_open_cv.h>
#include <opencv2/opencv.hpp> 

template <size_t _Rz, size_t _Cz> inline
void print_double_cvMat(CvMat *src) 
{
	for(int i=0; i<_Rz; i++)
	{
		for(int j=0; j<_Cz; j++)
		{
			std::cout << cvmGet(src, i, j) << " ";
		}
		std::cout << std::endl;
	}
}



template <size_t _Rz, size_t _Cz> inline
void double_matrix2cvMat(const dlib::matrix<double, _Rz, _Cz> &src, CvMat *dst) 
{
	cvZero(dst);
	for(int i=0; i<_Rz; i++)
	{
		for(int j=0; j<_Cz; j++) 
		{
			cvmSet(dst, i, j, src(i, j));
		}
	}		
}

template <size_t _Rz, size_t _Cz> inline
void double_cvMat2matrix(CvMat *src, dlib::matrix<double, _Rz, _Cz> &dst) 
{
	for(int i=0; i<_Rz; i++)
		for(int j=0; j<_Cz; j++)
			dst(i, j) = cvmGet(src, i, j);
}


/* 
 * Function: euler2matrix
 * Usage: dlib::matrix<T> R = euler2matrix(yaw, pitch, roll, false);
 * Parameters:
 * 		@yaw: Euler angle (radian);
 * 		@pitch: Euler angle (radian);
 * 		@roll: Euler angle (radian);
 * 		@is_linearized: Choose to use linearized Euler angle transform or not. If true, be sure yaw, pitch and roll
 * 						keep small.
 * Return:
 * 		Rotation matrix R.
 * 		If linearized:
 * 			R = [[1,      -yaw, pitch],
 * 				 [yaw,    1,    -roll],
 * 				 [-pitch, roll, 1    ]]
 * 		Else
 *			R = [[c2*c1, s3*s2*c1-c3*s1, c3*s2*c1+s3*s1],
 *				 [c2*s1, s3*s2*s1+c3*c1, c3*s2*s1-s3*c1],
 *				 [-s2,   s3*c2,          c3*c2         ]]; 
 *			(c1=cos(yaw),   s1=sin(yaw))
 *			(c2=cos(pitch), s2=sin(pitch))
 *			(c3=cos(roll),  s3=sin(roll))
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
		/* (Deprecated) Using angles. */ 
		// _Tp c1 = cos(yaw   * _Tp(M_PI) / _Tp(180.0)), s1 = sin(yaw   * _Tp(M_PI) / _Tp(180.0));
		// _Tp c2 = cos(pitch * _Tp(M_PI) / _Tp(180.0)), s2 = sin(pitch * _Tp(M_PI) / _Tp(180.0));
		// _Tp c3 = cos(roll  * _Tp(M_PI) / _Tp(180.0)), s3 = sin(roll  * _Tp(M_PI) / _Tp(180.0));
		
		/* Using radians */
		_Tp c1 = cos(yaw),   s1 = sin(yaw);
		_Tp c2 = cos(pitch), s2 = sin(pitch);
		_Tp c3 = cos(roll),  s3 = sin(roll);
		R = c2 * c1, s3 * s2 * c1 - c3 * s1, c3 * s2 * c1 + s3 * s1,
			c2 * s1, s3 * s2 * s1 + c3 * c1, c3 * s2 * s1 - s3 * c1,
			-s2,     s3 * c2,                c3 * c2; 
	}
	return R;
}


/* 
 * Function: points2homogeneous
 * Usage: dlib::matrix<T> quaternion_points = points2homogeneous(ternary_points);
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
dlib::matrix<_Tp> points2homogeneous(const dlib::matrix<_Tp> &ternary_points)
{
	dlib::matrix<_Tp> quaternion_points;
	quaternion_points.set_size(ternary_points.nr(), 4);
	dlib::set_subm(quaternion_points, dlib::range(0, ternary_points.nr()-1), dlib::range(0, 2)) = ternary_points;
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
dlib::matrix<_Tp> RT2P(const dlib::matrix<_Tp, 3, 3> &R, const dlib::matrix<_Tp, 3, 1> &T)
{
	dlib::matrix<_Tp> P;
	P.set_size(3, 4);
	dlib::set_subm(P, dlib::range(0, 2), dlib::range(0, 2)) = R;
	dlib::set_colm(P, 3) = T;
	return P;
}


template<typename _Tp, typename _Ep>
dlib::matrix<_Tp> transform_points(const dlib::matrix<_Tp, 3, 3> &R, const dlib::matrix<_Tp, 3, 1> &T, 
					     		   const dlib::matrix<_Ep> &points) 
{
	dlib::matrix<_Tp> P(3, 4);
	// std::cout << "origin: " << points(0) << " " << points(1) << " " << points(2) << std::endl;
    dlib::matrix<_Tp> tmp_points, before_points, after_points;
	P = RT2P(R, T);
	// std::cout << "P: " << P << std::endl;
	tmp_points = dlib::reshape(dlib::matrix_cast<_Tp>(points), points.nr()/3, 3);
	before_points = points2homogeneous(tmp_points);
	// std::cout << "quar: " << before_points(0, 0) << " " << before_points(1, 0) << " " << before_points(2, 0) << " " << before_points(3, 0) << std::endl;
	after_points = P * before_points;
    return dlib::reshape(dlib::trans(after_points), points.nr(), 1);
}


template<typename _Tp, typename _Ep>
dlib::matrix<_Tp> transform_points(const _Tp * const ext_parm, const dlib::matrix<_Ep> &points, 
								   bool is_linearized = false) 
{
	dlib::matrix<_Tp, 3, 3> R;
	dlib::matrix<_Tp, 3 ,1> T;

	const _Tp &yaw   = ext_parm[0];
	const _Tp &pitch = ext_parm[1];
	const _Tp &roll  = ext_parm[2];
	const _Tp &tx    = ext_parm[3];
	const _Tp &ty    = ext_parm[4];
	const _Tp &tz    = ext_parm[5];
	
	R = euler2matrix(yaw, pitch, roll, is_linearized);
	T = tx, ty, tz;
	return transform_points(R, T, points);
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

inline double mat_norm(dlib::matrix<double> &m)
{
	return dlib::sum(dlib::abs(m)) / (m.nr() * m.nc());
}


inline bool is_rotation_matrix(const dlib::matrix<double, 3, 3> &R)
{
    dlib::matrix<double> RRt = R * dlib::trans(R);
    dlib::matrix<double> I = dlib::identity_matrix<double>(3);
    dlib::matrix<double> dif = RRt - I;
    return (mat_norm(dif) < 1e-6);
}


inline void satisfy_extrinsic_matrix(dlib::matrix<double, 3, 3> &R, dlib::matrix<double, 3, 1> &T)
{
	CvMat *P_mat, *R_mat, *T_mat;
	dlib::matrix<double> P = RT2P(R, T);

	P_mat = cvCreateMat(3, 4, CV_64FC1);
	double_matrix2cvMat<3 ,4>(P, P_mat);
	
	// print_double_cvMat<3, 4>(P_mat);
	
	R_mat = cvCreateMatHeader(3, 3, CV_64FC1);
	cvGetCols(P_mat, R_mat, 0, 3);
	T_mat = cvCreateMatHeader(3, 1, CV_64FC1);
	cvGetCol(P_mat, T_mat, 3);

	if( cvDet(R_mat) < 0)
		cvScale(P_mat, P_mat, -1);
	double sc = cvNorm(R_mat);
	CV_Assert(fabs(sc) > DBL_EPSILON);

    double U[9], V[9], W[3];
    CvMat matU = cvMat(3, 3, CV_64F, U);
    CvMat matV = cvMat(3, 3, CV_64F, V);
    CvMat matW = cvMat(3, 1, CV_64F, W);

	cvSVD(R_mat, &matW, &matU, &matV, CV_SVD_MODIFY_A + CV_SVD_U_T + CV_SVD_V_T);
	cvGEMM(&matU, &matV, 1, 0, 0, R_mat, CV_GEMM_A_T);

	cvScale(T_mat, T_mat, cvNorm(R_mat)/sc);

	// print_double_cvMat<3, 3>(R_mat);
	// print_double_cvMat<3, 1>(T_mat);

	double_cvMat2matrix<3, 3>(R_mat, R);
	double_cvMat2matrix<3, 1>(T_mat, T);
}

