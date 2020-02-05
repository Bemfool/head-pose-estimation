#include "hpe.h"
#include <chrono>

using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;
using namespace dlib;
using namespace std;

int main(int argc, char** argv)
{  
	google::InitGoogleLogging(argv[0]);
	hpe hpe_problem("/home/keith/Desktop/head-pose-estimation/cpp/inputs.txt");
    
	// 打开图片获得人脸框选
	array2d<rgb_pixel> img;
	std::string img_name = "test2.jpg";

	if(argc > 1) img_name = argv[1];	
	std::cout << "processing image " << img_name << std::endl;
	load_image(img, img_name);

	image_window win;
	win.clear_overlay();
	double fx = hpe_problem.get_model().get_fx(), fy = hpe_problem.get_model().get_fy();
	double cx = hpe_problem.get_model().get_cx(), cy = hpe_problem.get_model().get_cy();

	try {
		// 初始化detector
		std::cout << "initing detector..." << std::endl;
		dlib::frontal_face_detector detector = get_frontal_face_detector();
		dlib::shape_predictor sp;
		deserialize("../data/shape_predictor_68_face_landmarks.dat") >> sp;
		std::cout << "detector init successfully\n" << std::endl;

		pyramid_up(img);
		std::vector<dlib::rectangle> dets = detector(img);
		std::cout << "Number of faces detected: " << dets.size() << std::endl;
		std::vector<dlib::full_object_detection> obj_detections;
		win.set_image(img);
		for (unsigned long j = 0; j < dets.size(); ++j) {
			// 将当前获得的特征点数据放置到全局
			full_object_detection obj_detection = sp(img, dets[j]);
			obj_detections.push_back(obj_detection);
			hpe_problem.set_observed_points(obj_detection);

            {
                std::cout << "* Solve extrinsic parameters using direct euler angles." << std::endl;
                auto start = std::chrono::system_clock::now();
                hpe_problem.solve_ext_params(USE_CERES);
                auto end = std::chrono::system_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                std::cout << "  Cost: " << double(duration.count())
                    * std::chrono::microseconds::period::num
                    / std::chrono::microseconds::period::den << " Seconds" << std::endl;                
                hpe_problem.get_model().clear_ext_params();
            }

            {
                std::cout << "* Solve extrinsic parameters using linearized euler angles (w/ DLT)." << std::endl;
                auto start = std::chrono::system_clock::now();
                hpe_problem.solve_ext_params(USE_CERES | USE_LINEARIZED_RADIANS | USE_DLT);
                auto end = std::chrono::system_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                std::cout << "  Cost: " << double(duration.count())
                    * std::chrono::microseconds::period::num
                    / std::chrono::microseconds::period::den << " Seconds" << std::endl;                
                hpe_problem.get_model().clear_ext_params();
            }
            
            {
                std::cout << "* Solve extrinsic parameters using linearized euler angles (w/o DLT)." << std::endl;
                auto start = std::chrono::system_clock::now();
                hpe_problem.solve_ext_params(USE_CERES | USE_LINEARIZED_RADIANS);
                auto end = std::chrono::system_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                std::cout << "  Cost: " << double(duration.count())
                    * std::chrono::microseconds::period::num
                    / std::chrono::microseconds::period::den << " Seconds" << std::endl;                
                hpe_problem.get_model().clear_ext_params();
            }

            {
                std::cout << "* Solve extrinsic parameters using OpenCV." << std::endl;
                auto start = std::chrono::system_clock::now();
                hpe_problem.solve_ext_params(USE_OPENCV);
                auto end = std::chrono::system_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                std::cout << "  Cost: " << double(duration.count())
                    * std::chrono::microseconds::period::num
                    / std::chrono::microseconds::period::den << " Seconds" << std::endl;                
                hpe_problem.get_model().clear_ext_params();
            }

            {
                std::cout << "* Solve shape coefficients." << std::endl;
                auto start = std::chrono::system_clock::now();
                hpe_problem.solve_shape_coef();
                auto end = std::chrono::system_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                std::cout << "  Cost: " << double(duration.count())
                    * std::chrono::microseconds::period::num
                    / std::chrono::microseconds::period::den << " Seconds" << std::endl;                
                hpe_problem.get_model().clear_ext_params();
            }

            {
                std::cout << "* Solve expression coefficients." << std::endl;
                auto start = std::chrono::system_clock::now();
                hpe_problem.solve_expr_coef();
                auto end = std::chrono::system_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                std::cout << "  Cost: " << double(duration.count())
                    * std::chrono::microseconds::period::num
                    / std::chrono::microseconds::period::den << " Seconds" << std::endl;                
                hpe_problem.get_model().clear_ext_params();
            }
		}
	} catch (exception& e) {
		cout << "\nexception thrown!" << endl;
		cout << e.what() << endl;
	}
}