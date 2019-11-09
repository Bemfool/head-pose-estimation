#include "hpe.h"

using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;
using namespace dlib;
using namespace std;

std::vector<point3f> final_landmarks3d(LANDMARK_NUM);	// 标准模型的特征点三维坐标
array2d<rgb_pixel> img;


int main(int argc, char** argv)
{  
	if(!init_bfm())
		return 0;
	google::InitGoogleLogging(argv[0]);
	image_window win;
	win.clear_overlay();
	try {
		// 初始化detector
		cout << "initing detector..." << endl;
		frontal_face_detector detector = get_frontal_face_detector();
		shape_predictor sp;
		deserialize("shape_predictor_68_face_landmarks.dat") >> sp;
		cout << "detector init successfully\n" << endl;

		// 打开图片获得人脸框选
		// cout << "processing image " << "2019-07-21-123713.jpg" << endl;
		// load_image(img, "2019-07-21-123713.jpg");
		cout << "processing image " << "2019-07-23-214644.jpg" << endl;
		load_image(img, "2019-07-23-214644.jpg");

		pyramid_up(img);
		std::vector<dlib::rectangle> dets = detector(img);
		cout << "Number of faces detected: " << dets.size() << endl;
		std::vector<dlib::full_object_detection> obj_detections;
		win.set_image(img);
		for (unsigned long j = 0; j < dets.size(); ++j) {
			// 将当前获得的特征点数据放置到全局
			full_object_detection obj_detection = sp(img, dets[j]);
			obj_detections.push_back(obj_detection);

			// 变量定义以及初始化
			double head_pose[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
			double shape_pc_basis[N_PC] = {0.f};

			Problem head_pose_problem, shape_pc_problem;
			CostFunction* head_pose_cost_function =
				new NumericDiffCostFunction<HeadPoseNumericCostFunctor, ceres::RIDDERS, LANDMARK_NUM, 6>(new HeadPoseNumericCostFunctor(obj_detection));
			head_pose_problem.AddResidualBlock(head_pose_cost_function, NULL, head_pose);
			CostFunction* shape_pc_cost_function = 
				new NumericDiffCostFunction<ShapeNumericCostFunctor, ceres::RIDDERS, LANDMARK_NUM, N_PC>(new ShapeNumericCostFunctor(obj_detection));
			shape_pc_problem.AddResidualBlock(shape_pc_cost_function, NULL, shape_pc_basis);
			Solver::Options options;
			options.minimizer_progress_to_stdout = true;
			Solver::Summary head_pose_summary, shape_pc_summary;

			std::cout << "begin solving ..." << std::endl;
			while(true) {
				Solve(options, &head_pose_problem, &head_pose_summary);
				std::cout << head_pose_summary.BriefReport() << endl;
				std::cout << "x : " << head_pose[0] << " " << head_pose[1] << " " << head_pose[2] << 
								" " << head_pose[3] << " " << head_pose[4] << " " << head_pose[5] << endl;	
				set_head_pose_parameters(head_pose);
				// getchar();

				Solve(options, &shape_pc_problem, &shape_pc_summary);
				std::cout << shape_pc_summary.BriefReport() << endl;
				set_shape_pc_basis(shape_pc_basis);
				// getchar();

				if(abs(shape_pc_summary.final_cost - head_pose_summary.final_cost) < 1)
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

		// 任意键退出
		cin.get();
		cin.get();
		cin.get();
		cin.get();
	} catch (exception& e) {
		cout << "\nexception thrown!" << endl;
		cout << e.what() << endl;
	}
}