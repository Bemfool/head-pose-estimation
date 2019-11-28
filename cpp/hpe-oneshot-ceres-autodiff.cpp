#include "functor.h"

using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;
using namespace dlib;
using namespace std;

std::vector<point3f> final_landmarks3d(LANDMARK_NUM);	// 标准模型的特征点三维坐标

int main(int argc, char** argv)
{  
	if(!init_bfm())
		return 0;

	// 打开图片获得人脸框选
	array2d<rgb_pixel> img;
	std::string img_name = "test.jpg";
	if(argc > 1) img_name = argv[1];	
	std::cout << "processing image " << img_name << std::endl;
	load_image(img, img_name);

	image_window win;
	win.clear_overlay();

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

			// 变量定义以及初始化
			double head_pose[7] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
			solve_head_pose(obj_detection, head_pose);
            
			get_landmarks(final_landmarks3d);
			transform(final_landmarks3d, head_pose);
			std::vector<point2d> final_landmarks2d;
			landmarks_3d_to_2d(PINHOLE, final_landmarks3d, final_landmarks2d);
			std::vector<dlib::full_object_detection> final_obj_detection;
			std::vector<point2d> parts;
			for(int i=0; i<LANDMARK_NUM; i++)
				parts.push_back(final_landmarks2d.at(i));
			final_obj_detection.push_back(dlib::full_object_detection(dlib::rectangle(), parts));
			win.add_overlay(render_face_detections(final_obj_detection));	
		}
		
		win.add_overlay(render_face_detections(obj_detections, rgb_pixel(0,0, 255)));
		std::cout << shape_coef << std::endl;

		cin.get();
	} catch (exception& e) {
		cout << "\nexception thrown!" << endl;
		cout << e.what() << endl;
	}
}