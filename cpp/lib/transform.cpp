#include "transform.h"


dlib::matrix<double> matrix_transform(dlib::matrix<double> R, double tx, double ty, double tz, 
					      const dlib::matrix<double> &points) {
	dlib::matrix<double> before_points(points);
    dlib::matrix<double> after_points;
	before_points = dlib::reshape(before_points, points.nr() / 3, 3);
	after_points = R * dlib::trans(before_points);
	dlib::set_rowm(after_points, 0) = dlib::rowm(after_points, 0) + tx;
	dlib::set_rowm(after_points, 1) = dlib::rowm(after_points, 1) + ty;
	dlib::set_rowm(after_points, 2) = dlib::rowm(after_points, 2) + tz;
	after_points = dlib::reshape(dlib::trans(after_points), points.nr(), 1);	
    return after_points;
}


bool is_rotation_matrix(const dlib::matrix<double, 3, 3> &R) {
    dlib::matrix<double> RRt = R * dlib::trans(R);
    dlib::matrix<double> I = dlib::identity_matrix<double>(3);
    dlib::matrix<double> dif = RRt - I;
    return (mat_norm(dif) < 1e-6);
}


void satisfy_rotation_constraint(dlib::matrix<double, 3, 3> &R) {
	R = dlib::pow(R * dlib::trans(R), 0.5) * R;
}


double mat_norm(dlib::matrix<double> &m) {
    return dlib::sum(dlib::abs(m)) / (m.nr() * m.nc());
}