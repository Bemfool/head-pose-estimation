#include "transform.h"

void landmarks_3d_to_2d(camera_type type, std::vector<point3f>& landmarks_3d, std::vector<point2d>& landmarks_2d)
{
	landmarks_2d.clear();
	double xs, ys, zs;
	double xo, yo;
	for(std::vector<point3f>::iterator iter=landmarks_3d.begin(); iter!=landmarks_3d.end(); ++iter) {
		xs = (*iter).x();
		ys = (*iter).y();
		zs = (*iter).z();
		if(type == PINHOLE) {
			xo = FX * xs / zs + CX;
			yo = FY * ys / zs + CY;
		} else if(type == PARALLEL) {
			xo = xs;
			yo = ys;
		} else {
			std::cout << "[ERROR] Projection from 3d to 2d failed." << std::endl;
		}
		landmarks_2d.push_back(point2d(xo, yo));
	}
}


void transform(std::vector<point3f>& points, const double * const x)
{
	rotate(points, x[0], x[1] ,x[2]);
	translate(points, x[3], x[4], x[5]);
}


void rotate(std::vector<point3f>& points, const double yaw, const double pitch, const double roll) 
{
	dlib::point_transform_affine3d around_z = dlib::rotate_around_z(roll * M_PI / 180);
	dlib::point_transform_affine3d around_y = dlib::rotate_around_y(yaw * M_PI / 180);
	dlib::point_transform_affine3d around_x = dlib::rotate_around_x(pitch * M_PI / 180);
	for(std::vector<point3f>::iterator iter=points.begin(); iter!=points.end(); ++iter)
		*iter = around_z(around_y(around_x(*iter)));
}


void translate(std::vector<point3f>& points, const double x, const double y, const double z)
{
	for(std::vector<point3f>::iterator iter=points.begin(); iter!=points.end(); ++iter)
		*iter = (*iter) + point3f(x, y, z);
}
