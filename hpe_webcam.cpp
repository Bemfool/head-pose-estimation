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

#include "hpe_problem.h"

#include <chrono>
#include <getopt.h>


const int N_MAX_RECORDS = 50;

const std::string INPUT_FILE_PATH = "/home/bemfoo/Project/head-pose-estimation/example_inputs/example_inputs_68.txt";
const std::string DEFAULT_VIDEO_PATH = "/home/bemfoo/Project/head-pose-estimation/data/person0.avi";
const std::string BACKGROUND_IMG_PATH = "/home/bemfoo/Project/head-pose-estimation/data/blank.jpg";
const std::string DLIB_LANDMARK_DETECTOR_DATA_PATH = "/home/bemfoo/Data/shape_predictor_68_face_landmarks.dat";

const unsigned int SCR_WIDTH = 640;
const unsigned int SCR_HEIGHT = 480;


int main(int argc, char** argv)
{  
	google::InitGoogleLogging(argv[0]);

	// Init head pose estimation problem 
	HeadPoseEstimationProblem *pHpeProblem = new HeadPoseEstimationProblem(INPUT_FILE_PATH);
	BaselFaceModelManager *pBfmManager = pHpeProblem->getModel();

	dlib::image_window win;
	cv::VideoCapture cap;

	// Parse command parameters
	int opt;
    static struct option cmdOptions[] = {
        {"video", optional_argument, nullptr, 'v'},
		{"help", no_argument, nullptr, 'h'},
        {0, 0, 0, 0} 
    };
    while ((opt = getopt_long(argc, argv, "v:", cmdOptions, nullptr)) != -1) {
       switch(opt)
	   {
			case 'v':
				if(optarg != nullptr)
				{
					BFM_DEBUG("Input local video: %s\n", optarg);
					cap.open(optarg);
				}
				else
				{
					BFM_DEBUG("Input built-in front camera\n");
					cap.open(0);
				}
				break;
			case 'h':
				BFM_DEBUG("Usage: hpe_webcam [OPTION] ... \n");
				BFM_DEBUG("Estimate head pose for given video stream.\n");
				BFM_DEBUG("\t-v\t--video[=FILE]\tLoad video stream. FILE can be video path (use built-in camera if omitted)\n");
				BFM_DEBUG("\t-h\t--help\tDisplay this help and exit\n");
				return 0;
			default:
				BFM_ERROR("Unrecognized command parameters\n");
				return -1;
	   }
    }

	if (!cap.isOpened()) {
		BFM_ERROR("Unable to connect to camera\n");
		BFM_DEBUG("Try to load default video path: %s\n", DEFAULT_VIDEO_PATH.c_str());
		cap.open(DEFAULT_VIDEO_PATH);
		if(!cap.isOpened())
		{
			BFM_ERROR("Still cannot open. Exit.\n");
			return -1;
		}
		else
		{
			BFM_DEBUG("Open successfully. Continue.\n");
		}
	}		

	dlib::array2d<dlib::rgb_pixel> arr2dBlankImg;
	load_image(arr2dBlankImg, BACKGROUND_IMG_PATH);
	
	try {
		// Init Detector
		dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
		dlib::shape_predictor sp;
		dlib::deserialize(DLIB_LANDMARK_DETECTOR_DATA_PATH) >> sp;
		BFM_DEBUG("Dlib landmark detector load path: %s\n", DLIB_LANDMARK_DETECTOR_DATA_PATH.c_str());

		double dFx = pBfmManager->getFx(), dFy = pBfmManager->getFy();
		double dCx = pBfmManager->getCx(), dCy = pBfmManager->getCy();

		bool bIsFirstFrame = true;

		auto timeFrameStart = std::chrono::system_clock::now();
		auto timeFrameEnd = std::chrono::system_clock::now();

		while(!win.is_closed()) {
			timeFrameEnd = std::chrono::system_clock::now();
			auto timeFrameDuration = std::chrono::duration_cast<std::chrono::microseconds>(timeFrameEnd - timeFrameStart);
			BFM_DEBUG("Cost of frame: %lf Second\n", 
				double(timeFrameDuration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den);             
			timeFrameStart = timeFrameEnd;

			cv::Mat matCurrentCapture;
			if(!cap.read(matCurrentCapture))
				break;
			dlib::cv_image<dlib::bgr_pixel> imgCurrentCapture(matCurrentCapture);

			/* This type image could not use pyramid_up()， and also I think for talking-head 
			 * video, this is unneccessary.
			 */
			// pyramid_up(imgCurrentCapture);

			std::vector<dlib::rectangle> aDets = detector(imgCurrentCapture);
			dlib::full_object_detection objDetection = sp(imgCurrentCapture, aDets[0]);

			if(aDets.size() != 0) {
				pHpeProblem->setObservedPoints(&objDetection);

				bool bHasConverged;
				if(bIsFirstFrame)
				{
					auto timeStart = std::chrono::system_clock::now();
					bHasConverged = false;
					while(!bHasConverged)
					{
						bHasConverged = pHpeProblem->solveExtParams(SolveExtParamsMode_UseCeres | SolveExtParamsMode_UseDlt | SolveExtParamsMode_UseLinearizedRadians);
						bHasConverged &= pHpeProblem->solveShapeCoef();
						bHasConverged &= pHpeProblem->solveExprCoef();
					}
					auto timeEnd = std::chrono::system_clock::now();
                	auto timeDuration = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart);
					BFM_DEBUG("Solution cost of first frame: %lf Second\n", 
						double(timeDuration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den);             
					bIsFirstFrame = false;
				}
				else
				{
					auto timeStart = std::chrono::system_clock::now();
					bHasConverged = false;
					while(!bHasConverged)
					{
						bHasConverged = pHpeProblem->solveExtParams(SolveExtParamsMode_UseCeres | SolveExtParamsMode_UseLinearizedRadians);
						bHasConverged &= pHpeProblem->solveExprCoef();
					}
					auto timeEnd = std::chrono::system_clock::now();
                	auto timeDuration = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart);
					BFM_DEBUG("Solution cost of frame: %lf Second\n", 
						double(timeDuration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den);             
					
				}

				unsigned int nLandmarks = pBfmManager->getNLandmarks();
				if(nLandmarks == N_DLIB_LANDMARK)
				{
					const Eigen::VectorXd vecLandmarkTransformed = pBfmManager->getLandmarkCurrentBlendshapeTransformed();
					std::vector<dlib::point> aPoints;

					for(unsigned int iLandmark = 0; iLandmark < nLandmarks; iLandmark++) {
						int u = int(dFx * vecLandmarkTransformed(iLandmark * 3) / vecLandmarkTransformed(iLandmark * 3 + 2) + dCx);
						int v = int(dFy * vecLandmarkTransformed(iLandmark * 3 + 1) / vecLandmarkTransformed(iLandmark * 3 + 2) + dCy);
						aPoints.push_back(dlib::point(u, v));
					}
					win.clear_overlay();
					win.add_overlay(render_face_detections(dlib::full_object_detection(dlib::rectangle(), aPoints)));
				}
			}
			else
			{
				win.clear_overlay();
			}
			
			// Select blank background or catured image
			win.set_image(arr2dBlankImg);
			// win.set_image(imgCurrentCapture); 
			win.add_overlay(render_face_detections(objDetection, dlib::rgb_pixel(0,0, 255)));

		}
		// Press any key to exit
		// std::cin.get();
	} catch (exception& e) {
		BFM_ERROR("Exception thrown: %s\n", e.what());
	}
}
