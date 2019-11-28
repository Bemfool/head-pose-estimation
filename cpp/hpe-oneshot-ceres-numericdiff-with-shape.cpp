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
			double head_pose[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
			double shape_pc_basis[N_ID_PC] = {0.f};
			double expr_pc_basis[N_EXPR_PC] = {0.f};

			Problem head_pose_problem, shape_pc_problem, expr_pc_problem;
			CostFunction* head_pose_cost_function =
				new NumericDiffCostFunction<HeadPoseNumericCostFunctor, ceres::RIDDERS, LANDMARK_NUM, 6>(new HeadPoseNumericCostFunctor(obj_detection));
			head_pose_problem.AddResidualBlock(head_pose_cost_function, NULL, head_pose);
			CostFunction* shape_pc_cost_function = 
				new NumericDiffCostFunction<ShapeNumericCostFunctor, ceres::RIDDERS, LANDMARK_NUM, N_ID_PC>(new ShapeNumericCostFunctor(obj_detection));
			shape_pc_problem.AddResidualBlock(shape_pc_cost_function, NULL, shape_pc_basis);
			CostFunction* expr_pc_cost_function = 
				new NumericDiffCostFunction<ExprNumericCostFunctor, ceres::RIDDERS, LANDMARK_NUM, N_EXPR_PC>(new ExprNumericCostFunctor(obj_detection));
			expr_pc_problem.AddResidualBlock(expr_pc_cost_function, NULL, expr_pc_basis);		
			Solver::Options options;
			options.minimizer_progress_to_stdout = true;
			options.max_num_iterations = 100;
			options.num_threads = 8;
			Solver::Summary head_pose_summary, shape_pc_summary, expr_pc_summary;

			std::cout << "begin solving ..." << std::endl;
			int cnt = 0;
			while(true) {
				Solve(options, &head_pose_problem, &head_pose_summary);
				std::cout << head_pose_summary.BriefReport() << endl;
				std::cout << "x : " << head_pose[0] << " " << head_pose[1] << " " << head_pose[2] << 
								" " << head_pose[3] << " " << head_pose[4] << " " << head_pose[5] << endl;	
				set_head_pose_parameters(head_pose);

				Solve(options, &shape_pc_problem, &shape_pc_summary);
				std::cout << shape_pc_summary.BriefReport() << std::endl;
				set_shape_pc_basis(shape_pc_basis);

				ply_write("test" + to_string(cnt++) + "shape.ply");

				Solve(options, &expr_pc_problem, &expr_pc_summary);
				std::cout << expr_pc_summary.BriefReport() << std::endl;
				set_expr_pc_basis(expr_pc_basis);

				ply_write("test" + to_string(cnt++) + "expr.ply");

				if(shape_pc_summary.initial_cost == shape_pc_summary.final_cost 
				&& head_pose_summary.final_cost == head_pose_summary.initial_cost
				&& expr_pc_summary.initial_cost == expr_pc_summary.final_cost)
					break;
			}
			
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