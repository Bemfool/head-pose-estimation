/**********************************************************
 * Author: Keith Lin
 * Date: 2019-9-25
 * Description: This demo is used to fitting landmarks got from dlib with a standard
 * model, using Ceres as mathematic tool to deal with the unlinear optimisation. 
 * Since dlib's detector takes rather long time, users should cmake with AVX and SSE2
 * or SSE4, just like:
 * 		cmake .. -D USE_AVX_INSTRUCTIONS=ON
 * 		cmake .. -D USE_SSE2_INSTRUCTIONS=ON
 * 		cmake .. -D USE_SSE4_INSTRUCTIONS=ON
 * And do not forget to build Release version (NO DEBUG), or your video capturing
 * would become stuttering.
 **********************************************************/
#include "hpe.h"

using ceres::NumericDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;
using namespace dlib;
using namespace std;

/* 3d landmarks coordinates of standard model */
std::vector<point3f> model_landmarks(LANDMARK_NUM);

int main(int argc, char** argv)
{  
	image_window win;
	try {
		// Init Detector
		cout << "initing detector..." << endl;
		frontal_face_detector detector = get_frontal_face_detector();
		shape_predictor sp;
		deserialize("shape_predictor_68_face_landmarks.dat") >> sp;
		cout << "detector init successfully\n" << endl;

		// Read 3d landmarks coordinates of standard model from file
		cout << "loading model landmarks..." << endl;
		if(get_landmark(model_landmarks)) {
			cout << "model landmarks loaded successfully\n" << endl;
		} else {
			cout << "model landmarks loaded failed\n" << endl;
			return 0;
		}

		/* Use Camera */
		cv::VideoCapture cap(0);
        if (!cap.isOpened())
        {
            cerr << "Unable to connect to camera" << endl;
            return 1;
        }		

		/* Init variables */
		double x[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
		
		while(!win.is_closed()) {
			cv::Mat temp;
			if(!cap.read(temp))
				break;
			dlib::cv_image<bgr_pixel> img(temp);

			/* This type image could not use pyramid_up()， and also I think for talking-head 
			 * video, this is unneccessary.
			 */
			// pyramid_up(img);

			/* Use dlib to detect faces */
			std::vector<rectangle> dets = detector(img);
			cout << "Number of faces detected: " << dets.size() << endl;
	
			std::vector<full_object_detection> shapes;
			
			for (unsigned long j = 0; j < dets.size(); ++j) {
				/* Use dlib to get landmarks */
				full_object_detection shape = sp(img, dets[j]);

				coarse_estimation(x, shape, model_landmarks);

				/* Use Ceres to solve problem */
				Problem real_problem, coarse_problem;
				CostFunction* coarse_cost_function =
					new NumericDiffCostFunction<NumericCostFunctor, ceres::RIDDERS, LANDMARK_NUM, 6>(new NumericCostFunctor(shape, model_landmarks, COARSE));
				coarse_problem.AddResidualBlock(coarse_cost_function, NULL, x);

				// CostFunction* cost_function =
				// 	new NumericDiffCostFunction<CostFunctor, ceres::RIDDERS, LANDMARK_NUM, 6>(new CostFunctor(shape));
				CostFunction* real_cost_function =
					new NumericDiffCostFunction<NumericCostFunctor, ceres::RIDDERS, LANDMARK_NUM, 6>(new NumericCostFunctor(shape, model_landmarks, REAL));
				
				real_problem.AddResidualBlock(real_cost_function, NULL, x);
				Solver::Options options;
				/* [TEST] If video is stuttering, we need to make optimisation caurser */
				// options.max_num_iterations = 10;
				options.minimizer_progress_to_stdout = true;
				Solver::Summary summary;
				
				Solve(options, &coarse_problem, &summary);
				Solve(options, &real_problem, &summary);
				std::cout << summary.BriefReport() << endl;
				std::cout << "x : " << x[0] << " " << x[1] << " " << x[2] << " " << x[3] << " " << x[4] << " " << x[5] << endl;
				
				/* Display final result from fitting
				 * 		Green lines: fit landmarks
				 * 		Blue lines: dlib landmarks
				 */
				std::vector<point3f> fitting_landmarks;
				for(std::vector<point3f>::iterator iter=model_landmarks.begin(); iter!=model_landmarks.end(); ++iter)
					fitting_landmarks.push_back(*iter);
				transform(fitting_landmarks, x);
				std::vector<point> model_landmarks_2d;
				landmarks_3d_to_2d(REAL, fitting_landmarks, model_landmarks_2d);
				std::vector<full_object_detection> model_shapes;
				std::vector<point> parts;
				for(int i=0; i<LANDMARK_NUM; i++)
					parts.push_back(model_landmarks_2d.at(i));
				model_shapes.push_back(full_object_detection(rectangle(), parts));
				win.clear_overlay();
				win.add_overlay(render_face_detections(model_shapes));	
				shapes.push_back(shape);
			}
			/* To avoid the situation that landmarks gets stuck with no face detected */
			if(dets.size()==0)
				win.clear_overlay();

			win.set_image(img);
			win.add_overlay(render_face_detections(shapes, rgb_pixel(0,0, 255)));
	
		}
		// Press any key to exit
		cin.get();
	} catch (exception& e) {
		cout << "\nexception thrown!" << endl;
		cout << e.what() << endl;
	}
}
