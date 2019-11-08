#include "hpe.h"

using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;
using namespace dlib;
using namespace std;

full_object_detection current_shape;
std::vector<point3f> model_landmarks(LANDMARK_NUM);	// 标准模型的特征点三维坐标
std::vector<point3f> fitting_landmarks(LANDMARK_NUM); // 经过旋转、平移的特征点三维坐标
array2d<rgb_pixel> img;

int main(int argc, char** argv)
{  
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

		// 读取标准模型的特征点三维坐标
		cout << "loading model landmarks..." << endl;
		if(get_landmark(model_landmarks)) {
			cout << "model landmarks loaded successfully\n" << endl;
		} else {
			cout << "model landmarks loaded failed\n" << endl;
			return 0;
		}

		// 打开图片获得人脸框选
		// cout << "processing image " << "2019-07-21-123713.jpg" << endl;
		// load_image(img, "2019-07-21-123713.jpg");
		cout << "processing image " << "2019-07-23-214644.jpg" << endl;
		load_image(img, "2019-07-23-214644.jpg");


		pyramid_up(img);
		std::vector<rectangle> dets = detector(img);
		cout << "Number of faces detected: " << dets.size() << endl;
		std::vector<full_object_detection> shapes;
		win.set_image(img);
		for (unsigned long j = 0; j < dets.size(); ++j) {
			// 将当前获得的特征点数据放置到全局
			full_object_detection shape = sp(img, dets[j]);
			current_shape = shape;

			// 变量定义以及初始化
			double x[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

			coarse_estimation(x, shape, model_landmarks);

			Problem real_problem, coarse_problem;
			// CostFunction* cost_function =
			// 	new NumericDiffCostFunction<CostFunctor, ceres::RIDDERS, LANDMARK_NUM, 6>(new CostFunctor());
			CostFunction* coarse_cost_function =
				new NumericDiffCostFunction<NumericCostFunctor, ceres::RIDDERS, LANDMARK_NUM, 6>(new NumericCostFunctor(shape, model_landmarks, COARSE));
			CostFunction* real_cost_function =
				new NumericDiffCostFunction<NumericCostFunctor, ceres::RIDDERS, LANDMARK_NUM, 6>(new NumericCostFunctor(shape, model_landmarks, REAL));
			coarse_problem.AddResidualBlock(coarse_cost_function, NULL, x);
			real_problem.AddResidualBlock(real_cost_function, NULL, x);
			Solver::Options options;
			// options.gradient_tolerance = true;
			options.minimizer_progress_to_stdout = true;
			Solver::Summary summary;
			Solve(options, &coarse_problem, &summary);
			Solve(options, &real_problem, &summary);
			std::cout << summary.BriefReport() << endl;
			std::cout << "x : " << x[0] << " " << x[1] << " " << x[2] << " " << x[3] << " " << x[4] << " " << x[5] << endl;
			
			fitting_landmarks.clear();
			for(std::vector<point3f>::iterator iter=model_landmarks.begin(); iter!=model_landmarks.end(); ++iter)
				fitting_landmarks.push_back(*iter);
			transform(fitting_landmarks, x);
			std::vector<point> model_landmarks_2d;
			landmarks_3d_to_2d(PINHOLE, fitting_landmarks, model_landmarks_2d);
			std::vector<full_object_detection> model_shapes;
			std::vector<point> parts;
			for(int i=0; i<LANDMARK_NUM; i++)
				parts.push_back(model_landmarks_2d.at(i));
			model_shapes.push_back(full_object_detection(rectangle(), parts));
			win.add_overlay(render_face_detections(model_shapes));	
			shapes.push_back(shape);
		}
		
		// image_window dlib_win;
		// // 显示dlib获取的特征点
		// dlib_win.set_title("face poses");
		// dlib_win.set_image(img);
		win.add_overlay(render_face_detections(shapes, rgb_pixel(0,0, 255)));

		// 任意键退出
		cin.get();
	} catch (exception& e) {
		cout << "\nexception thrown!" << endl;
		cout << e.what() << endl;
	}
}