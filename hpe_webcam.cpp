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
#include <chrono>
using namespace dlib;

const int MAX_N_RECORD = 50;

// settings
const unsigned int SCR_WIDTH = 640;
const unsigned int SCR_HEIGHT = 480;

int main(int argc, char** argv)
{  
	google::InitGoogleLogging(argv[0]);
	hpe hpe_problem("/home/keith/head-pose-estimation/inputs.txt");
	dlib::image_window win;
	std::ofstream out;
	
	/* Use Camera */
	cv::VideoCapture cap;
	bool is_save = false;
	if(argc == 2) 
	{
		out.open(argv[1], std::ios::out);
	}
	else if(argc == 3)
	{
		if(argv[1] == "0")
		{
			cap.open(0);
		}
		else
		{
			is_save = true;
			cap.open(argv[1]);
			std::cout << "open video " << argv[1] << std::endl;
		}

		if(argv[2] != "0")
		{
			out.open(argv[2], std::ios::out);
			std::cout << "file name: " << argv[2] << std::endl;
		}
		else
			out.open("seq.txt", std::ios::out);
	} else
	{
		out.open("seq.txt", std::ios::out);
		cap.open(0);
	}	
		
	if (!out) 
	{
		std::cerr << "Creation of text file to be saved failed.\n";
		return -1;
	}

	array2d<rgb_pixel> blank_img;
	load_image(blank_img, "blank.png");
	
	try {
		// Init Detector
		std::cout << "initing detector..." << std::endl;
		dlib::frontal_face_detector detector = get_frontal_face_detector();
		dlib::shape_predictor sp;
		deserialize("../data/shape_predictor_68_face_landmarks.dat") >> sp;
		std::cout << "detector init successfully\n" << std::endl;

        if (!cap.isOpened()) {
            std::cerr << "Unable to connect to camera" << std::endl;
            return 1;
        }		

		double fx = hpe_problem.get_model().get_fx(), fy = hpe_problem.get_model().get_fy();
		double cx = hpe_problem.get_model().get_cx(), cy = hpe_problem.get_model().get_cy();

		bool is_first_frame = true;

		auto f_start = std::chrono::system_clock::now();
		auto f_end = std::chrono::system_clock::now();

		while(!win.is_closed()) {
			f_end = std::chrono::system_clock::now();
			auto f_duration = std::chrono::duration_cast<std::chrono::microseconds>(f_end - f_start);
			std::cout << "Frame cost: " << double(f_duration.count())
				* std::chrono::microseconds::period::num
				/ std::chrono::microseconds::period::den << " Seconds\n" << std::endl;                
			f_start = f_end;

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
			// cout << "Number of faces detected: " << dets.size() << endl;
	
			std::vector<dlib::full_object_detection> obj_detections;
			for (unsigned long j = 0; j < dets.size(); ++j) {
				// 将当前获得的特征点数据放置到全局
				full_object_detection obj_detection = sp(img, dets[j]);
				obj_detections.push_back(obj_detection);
				hpe_problem.set_observed_points(obj_detection);

				for(int i=0; i<obj_detection.num_parts(); i++) 
					out << obj_detection.part(i).x() << " " << obj_detection.part(i).y() << " "; 
				out << "\n";

				bool state;
				if(is_first_frame)
				{
					auto start = std::chrono::system_clock::now();
					state = false;
					while(!state)
					{
						state = hpe_problem.solve_ext_params(USE_LINEARIZED_RADIANS | USE_DLT);
						state &= hpe_problem.solve_shape_coef();
						state &= hpe_problem.solve_expr_coef();
					}
					auto end = std::chrono::system_clock::now();
                	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                	std::cout << "First frame cost: " << double(duration.count())
						* std::chrono::microseconds::period::num
                    	/ std::chrono::microseconds::period::den << " Seconds\n" << std::endl;                

					is_first_frame = false;
				}
				else
				{
					auto start = std::chrono::system_clock::now();
					state = false;
					while(!state)
					{
						state = hpe_problem.solve_ext_params(USE_LINEARIZED_RADIANS);
						state &= hpe_problem.solve_expr_coef();
					}
					auto end = std::chrono::system_clock::now();
                	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                	std::cout << "First frame cost: " << double(duration.count())
						* std::chrono::microseconds::period::num
                    	/ std::chrono::microseconds::period::den << " Seconds\n" << std::endl;                

				}
				// hpe_problem.get_model().write_ext_params_to_file(out);

				// hpe_problem.get_model().print_extrinsic_params();
				// hpe_problem.get_model().print_intrinsic_params();
				// hpe_problem.get_model().print_shape_coef();
				// hpe_problem.get_model().print_expr_coef();

				const dlib::matrix<double> _fp_shape = hpe_problem.get_model().get_fp_current_blendshape();
				const dlib::matrix<double> fp_shape = transform_points(
					hpe_problem.get_model().get_mutable_extrinsic_params(), _fp_shape);
				std::vector<point2d> parts;

				for(int i=0; i<hpe_problem.get_model().get_n_landmark(); i++) {
					int u = int(fx * fp_shape(i*3) / fp_shape(i*3+2) + cx);
					int v = int(fy * fp_shape(i*3+1) / fp_shape(i*3+2) + cy);
					parts.push_back(point2d(u, v));
				}
				std::vector<dlib::full_object_detection> final_obj_detection;
				final_obj_detection.push_back(dlib::full_object_detection(dlib::rectangle(), parts));
				win.clear_overlay();
				win.add_overlay(render_face_detections(final_obj_detection));

			}
			/* To avoid the situation that landmarks gets stuck with no face detected */
			if(dets.size()==0)
				win.clear_overlay();

			win.set_image(blank_img);
			// win.set_image(img);
			win.add_overlay(render_face_detections(obj_detections, rgb_pixel(0,0, 255)));

			double *ext_params = hpe_problem.get_model().get_mutable_extrinsic_params();
			std::cout << ext_params[0] << " " << ext_params[1] << " " << ext_params[2] << std::endl;
		}
		// Press any key to exit
		// std::cin.get();
		out.close();
	} catch (exception& e) {
		std::cout << "\nexception thrown!" << std::endl;
		std::cout << e.what() << std::endl;
	}
}
