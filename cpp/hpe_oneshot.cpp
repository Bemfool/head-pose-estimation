#include "hpe.h"

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

			std::cout << "solving external parameters..." << std::endl;
			// hpe_problem.estimate_ext_parm();

			if(argc > 2)
				hpe_problem.solve_ext_parm(USE_CERES & USE_LINEARIZED_RADIANS, atof(argv[2]), atof(argv[3]));
			else
				hpe_problem.solve_ext_parm(USE_CERES & USE_LINEARIZED_RADIANS);
				
			// hpe_problem.solve_ext_parm(USE_CERES);
			// std::cout << "solving shape coeficients..." << std::endl;
			// hpe_problem.solve_shape_coef();
			// std::cout << "solving expression coeficients..." << std::endl;
			// hpe_problem.solve_expr_coef();	
            hpe_problem.get_model().print_external_parm();
			hpe_problem.get_model().print_intrinsic_parm();
			// hpe_problem.get_model().print_shape_coef();
			// hpe_problem.get_model().print_expr_coef();
			hpe_problem.get_model().generate_face();
			hpe_problem.get_model().ply_write("rnd_face.ply", (CAMERA_COORD | PICK_FP));

			const dlib::matrix<double> fp_shape = hpe_problem.get_model().get_fp_current_blendshape_transformed();
			std::vector<point2d> parts;

			for(int i=0; i<hpe_problem.get_model().get_n_landmark(); i++) {
				int u = int(fx * fp_shape(i*3) / fp_shape(i*3+2) + cx);
				int v = int(fy * fp_shape(i*3+1) / fp_shape(i*3+2) + cy);
				parts.push_back(point2d(u, v));
			}
			std::vector<dlib::full_object_detection> final_obj_detection;
			final_obj_detection.push_back(dlib::full_object_detection(dlib::rectangle(), parts));
			win.add_overlay(render_face_detections(final_obj_detection));

		}
		win.add_overlay(render_face_detections(obj_detections, rgb_pixel(0, 0, 255)));

		cin.get();
	} catch (exception& e) {
		cout << "\nexception thrown!" << endl;
		cout << e.what() << endl;
	}
}