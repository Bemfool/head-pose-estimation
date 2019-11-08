#pragma once
#include <vector>
#include "vec.h"
#include "constant.h"
#include <dlib/opencv.h>
#include <cmath>

#define _USE_MATH_DEFINES

/* Function: landmarks_3d_to_2d
 * Usage: landmarks_3d_to_2d(landmarks_3d, landmarks_2d);
 * Parameters: 
 * 		landmarks_3d: 3d landmarks coordinates
 * 		landmarks_2d: transformed 2d landmarks coordinates to be saved
 * --------------------------------------------------------------------------------------------
 * Transform 3d landmarks into 2d landmarks.
 */

void landmarks_3d_to_2d(camera_type type, std::vector<point3f>& landmarks_3d, std::vector<point2d>& landmarks_2d);


/* Function: transform
 * Usage: transform(x);
 * Parameters:
 * 		points: 3d coordinates to be transform
 * 		x: a double array of length 6.
 * 			x[0]: yaw
 * 			x[1]: pitch
 * 			x[2]: roll
 * 			x[3]: tx
 * 			x[4]: ty
 * 			x[5]: tz
 * --------------------------------------------------------------------------------------------
 * Transform a series 3d points to rotate with R(yaw, pitch, roll) and translate with T(tx, ty, tz).
 * Actually, this function encapsulates rotate() and translate() two functions.
 */
void transform(std::vector<point3f>& points, const double * const x);


/* Function: rotate
 * Usage: rotate(points, yaw, pitch, roll);
 * Parameters:
 * 		points: 3d coordinates to be transform
 * 		yaw: angle to rotate with y axis
 * 		pitch: angle to rotate with x axis
 * 		roll: angle to rotate with z axis 
 * --------------------------------------------------------------------------------------------
 * Transform a series 3d points to rotate with R(yaw, pitch, roll)
 */
void rotate(std::vector<point3f>& points, const double yaw, const double pitch, const double roll);


/* Function: translate
 * Usage: translate(points, x, y, z);
 * Parameters:
 * 		points: 3d coordinates to be transform
 * 		x distance to translate along x axis
 * 		y: distance to translate along y axis
 * 		z: distance to translate along z axis
 * --------------------------------------------------------------------------------------------
 * Transform a series 3d points to translate with T(x, y, z)
 */
void translate(std::vector<point3f>& points, const double x, const double y, const double z);

