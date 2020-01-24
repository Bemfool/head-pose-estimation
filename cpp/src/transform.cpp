#include "transform.h"

bool is_rotation_matrix(const dlib::matrix<double, 3, 3> &R) 
{
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