#include "hpe_problem.h"

#include <chrono>
#include <string>


const std::string INPUT_FILE_PATH = "/home/bemfoo/Project/head-pose-estimation/example_inputs/example_inputs_68.txt";
// const std::string INPUT_FILE_PATH = "/home/bemfoo/Project/head-pose-estimation/example_inputs/example_inputs_6.txt";
// const std::string INPUT_FILE_PATH = "/home/bemfoo/Project/head-pose-estimation/example_inputs/example_inputs_12.txt";
const std::string DEFAULT_IMG_PATH = "/home/bemfoo/Project/head-pose-estimation/data/profile0.jpg";
const std::string DLIB_LANDMARK_DETECTOR_DATA_PATH = "/home/bemfoo/Data/shape_predictor_68_face_landmarks.dat";

int main(int argc, char** argv)
{  
	google::InitGoogleLogging(argv[0]);

	// Init head pose estimation problem 
	HeadPoseEstimationProblem *pHpeProblem = new HeadPoseEstimationProblem(INPUT_FILE_PATH);
	BaselFaceModelManager *pBfmManager = pHpeProblem->getModel();

	/* If use different input file, 
	 * please update corresponding parameters in include/db_params.h 
	 */
	// HeadPoseEstimationProblem pHpeProblem("../example_inputs/example_inputs_6.txt");

	dlib::array2d<dlib::rgb_pixel> arr2dImg;
	std::string strImgName = DEFAULT_IMG_PATH;
	if(argc > 1) strImgName = argv[1];	
	BFM_DEBUG("Image to be processed: %s\n", strImgName.c_str());
	load_image(arr2dImg, strImgName);	

	// Init window (Dlib support) 
	dlib::image_window win;
	win.clear_overlay();

	// Fetch intrinsic parameter
	double dFx = pBfmManager->getFx(), dFy = pBfmManager->getFy();
	double dCx = pBfmManager->getCx(), dCy = pBfmManager->getCy();

	try 
	{
		// Init detector
		dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
		dlib::shape_predictor sp;
		dlib::deserialize(DLIB_LANDMARK_DETECTOR_DATA_PATH) >> sp;
		BFM_DEBUG("Dlib landmark detector load path: %s\n", DLIB_LANDMARK_DETECTOR_DATA_PATH.c_str());

		// pyramid_up(img);
		std::vector<dlib::rectangle> aDets = detector(arr2dImg);
		std::cout << "Number of faces detected: " << aDets.size() << std::endl;
		std::vector<dlib::full_object_detection> aObjDetections;
		win.set_image(arr2dImg);

		/* Only detect the first face */
		if(aDets.size() != 0) 
		{
			dlib::full_object_detection objDetection = sp(arr2dImg, aDets[0]);
			aObjDetections.push_back(objDetection);
			pHpeProblem->setObservedPoints(&objDetection);

			// Start of solving
			auto timeStart = std::chrono::system_clock::now();

			// There are some different ways to choose to solve external parameters 
			if(argc > 2)
				pHpeProblem->solveExtParams(
					SolveExtParamsMode_UseCeres | SolveExtParamsMode_UseDlt | SolveExtParamsMode_UseLinearizedRadians, 
					atof(argv[2]), atof(argv[3]));
			else
				pHpeProblem->solveExtParams(SolveExtParamsMode_UseCeres | SolveExtParamsMode_UseDlt | SolveExtParamsMode_UseLinearizedRadians);
			// pHpeProblem->solveExtParams(SolveExtParamsMode_UseOpenCV);

			pHpeProblem->solveShapeCoef();
			pHpeProblem->solveExprCoef();
			
			// End of solving
			auto timeEnd = std::chrono::system_clock::now();
			auto timeDuration = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart);
			BFM_DEBUG("Cost of solution: %lf Second\n", 
				double(timeDuration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den);             
            pBfmManager->printExtParams();

			// pBfmManager->printShapeCoef();
			// pBfmManager->printExprCoef();

			pBfmManager->genFace();
			pBfmManager->writePly("rnd_face.ply", (ModelWriteMode_CameraCoord | ModelWriteMode_PickLandmark));

			// Use dlib API to draw landmarks, only draw when number of landmark equals to 68 
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
				win.add_overlay(render_face_detections(dlib::full_object_detection(dlib::rectangle(), aPoints)));
			}
		}

		win.add_overlay(render_face_detections(aObjDetections, dlib::rgb_pixel(0, 0, 255)));
		cin.get();

	} catch (exception& e) {
		BFM_ERROR("Exception thrown: %s\n", e.what());
	}

}