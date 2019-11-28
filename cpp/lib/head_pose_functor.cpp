#include "head_pose_functor.h"

HeadPoseNumericCostFunctor::HeadPoseNumericCostFunctor(full_object_detection _shape) : shape(_shape) {} 

bool HeadPoseNumericCostFunctor::operator()(const double* const x, double* residual) const {
	/* Init landmarks to be transformed */
	std::vector<point3f> landmarks3d(LANDMARK_NUM);
	get_landmarks(landmarks3d);
	std::vector<point2d> landmarks2d;	
	
	double yaw   = x[0];
	double pitch = x[1];
	double roll  = x[2];
	double tx    = x[3];
	double ty    = x[4];
	double tz    = x[5];

	/* We use Z1Y2X3 format of Tait–Bryan angles */
	double c1 = cos(roll  * pi / 180.0), s1 = sin(roll  * pi / 180.0);
	double c2 = cos(yaw   * pi / 180.0), s2 = sin(yaw   * pi / 180.0);
	double c3 = cos(pitch * pi / 180.0), s3 = sin(pitch * pi / 180.0);

	for(std::vector<point3f>::iterator iter=landmarks3d.begin(); iter!=landmarks3d.end(); ++iter) {
		double X = iter->x(), Y = iter->y(), Z = iter->z(); 
		iter->x() = ( c1 * c2) * X + (c1 * s2 * s3 - c3 * s1) * Y + ( s1 * s3 + c1 * c3 * s2) * Z + tx;
		iter->y() = ( c2 * s1) * X + (c1 * c3 + s1 * s2 * s3) * Y + (-c3 * s1 * s2 - c1 * s3) * Z + ty;
		iter->z() = (-s2     ) * X + (c2 * s3               ) * Y + ( c2 * c3               ) * Z + tz; 
	}
	
	landmarks_3d_to_2d(PINHOLE, landmarks3d, landmarks2d);

	/* Calculate the energe (Euclid distance from two points) */
	for(int i=0; i<LANDMARK_NUM; i++) {
		long tmp1 = shape.part(i).x() - landmarks2d.at(i).x();
		long tmp2 = shape.part(i).y() - landmarks2d.at(i).y();
		residual[i] = sqrt(tmp1 * tmp1 + tmp2 * tmp2);
	}
	return true;
}
