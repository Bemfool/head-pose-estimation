#ifndef HPE_PROBLEM_H
#define HPE_PROBLEM_H

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <assert.h>
#include "ceres/ceres.h"
#include "string_utils.h"
#include "io_utils.h"
#include "functor/functor.h"


using FullObjectDetection = dlib::full_object_detection;
using Eigen::Matrix;
using Eigen::Vector3d;
using Eigen::VectorXd;


#define CERES_INIT(N_ITERATIONS, N_THREADS, B_STDCOUT) \
		ceres::Solver::Options options; \
		options.max_num_iterations = N_ITERATIONS; \
 		options.num_threads = N_THREADS; \
		options.minimizer_progress_to_stdout = B_STDCOUT; \
		ceres::Solver::Summary summary;


class HeadPoseEstimationProblem {


public:


	/* Initialization 
	 * Usage:
	 *     HeadPoseEstimationProblem hpe_problem;
	 *     HeadPoseEstimationProblem hpe_problem(filename);
	 * Parameters:
	 * 	   @filename: Filename for Basel Face Model loader.
	 *******************************************************************************
	 * Init a head pose estimation problem with input filename as bfm input filename.
	 */

	HeadPoseEstimationProblem() {}
	HeadPoseEstimationProblem(std::string filename);
	BfmStatus init(std::string filename);


	inline BaselFaceModelManager *getModel() { return m_pModel; }

	
	inline void setObservedPoints(FullObjectDetection *pObservedPoints) { m_pObservedPoints = pObservedPoints; }

	/*
	 * Function: solveExtParams
	 * Usage: hpe_problem.solveExtParams(FLAG1 | FLAG2 | ...);
	 * Parameters:
	 * 		@mode: Solve mode:
	 * 			- USE_CERES: Using Ceres Solver to solve, using euler angles;
	 * 			- USE_OpenCV: Using OpenCV to solve;
	 * 			- USE_DLT: Using DLT algorithm to estimate intial values. If using
	 * 		      OpenCV or linearized euler angles, it is ON by default.
	 * 			- USE_LINEARIZED_RADIANS: Using linearized euler angle to solve. If using
	 * 			  Ceres Solver, it is OFF by default.
	 * 			This flags can be combined.
	 * 		@ca: Norm coefficient of rotation parameters;
	 * 		@cb: Norm coefficient of translation parameters.
	 *******************************************************************************
	 * Estimate extrinsic parameters using reprojection. And final result will be set
	 * in model.
	 * 
	 */

	bool solveExtParams(long mode = SolveExtParamsMode_UseCeres, double ca = 1.0, double cb = 0.0);


	bool solveShapeCoef();


	bool solveExprCoef();


private:


	void dlt();


	bool is_close_enough(double *ext_params, double rotation_eps = 0, double translation_eps = 0);
	

	FullObjectDetection *m_pObservedPoints;
	BaselFaceModelManager *m_pModel;
	std::vector<unsigned int> m_aLandmarkMap;

};


#endif // HPE_PROBLEM_H