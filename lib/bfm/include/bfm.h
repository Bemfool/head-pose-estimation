#pragma once
#include "data.h"
#include "random.h"
#include "transform.h"
#include "type_utils.h"

class bfm {
public:
	bfm() {}
	bfm(const std::string filename);
	void init(const std::string filename);
	void data_check() const;
	void generate_random_face(double scale = 0.0);
	void generate_random_face(double shape_scale, double tex_scale, double expr_scale);
	void generate_average_face() { generate_random_face(0.0); }
	void generate_face();
	void generate_fp_face();
	template<typename T>
	dlib::matrix<T> generate_fp_face(const T * const shape_coef_,const T * const expr_coef_) const 
	{
		dlib::matrix<T> fp_current_shape_ = coef2object(shape_coef_, fp_shape_mu, fp_shape_pc, shape_ev, n_id_pc);
		dlib::matrix<T> fp_current_expr_ = coef2object(expr_coef_, fp_expr_mu, fp_expr_pc, expr_ev, n_expr_pc);
		dlib::matrix<T> fp_current_blendshape_ = fp_current_shape_ + fp_current_expr_;	
		return fp_current_blendshape_;
	}
	template<typename T>
	dlib::matrix<T> generate_fp_face_by_shape(const T * const shape_coef_) const 
	{
		dlib::matrix<T> fp_current_shape_ = coef2object(shape_coef_, fp_shape_mu, fp_shape_pc, shape_ev, n_id_pc);
		dlib::matrix<T> fp_current_expr_ = dlib::matrix_cast<T>(fp_current_expr);
		dlib::matrix<T> fp_current_blendshape_ = fp_current_shape_ + fp_current_expr_;	
		return fp_current_blendshape_;		
	}
	template<typename T>
	dlib::matrix<T> generate_fp_face_by_expr(const T * const expr_coef_) const 
	{
		dlib::matrix<T> fp_current_shape_ = dlib::matrix_cast<T>(fp_current_shape);
		dlib::matrix<T> fp_current_expr_ = coef2object(expr_coef_, fp_expr_mu, fp_expr_pc, expr_ev, n_expr_pc);
		dlib::matrix<T> fp_current_blendshape_ = fp_current_shape_ + fp_current_expr_;	
		return fp_current_blendshape_;		
	}
	void generate_transform_matrix();
	void generate_rotation_matrix();
	void generate_translation_vector();
	void generate_external_parameter();
	void accumulate_extrinsic_params(double *x);
	void ply_write(std::string fn = "face.ply", long mode = NONE_MODE) const;
	void ply_write_fp(std::string fn = "fp_face.ply") const;

	const int get_n_id_pc() const { return n_id_pc; }
	const int get_n_expr_pc() const { return n_expr_pc; }
	const int get_n_face() const { return n_face; }
	const int get_n_vertice() const { return n_vertice; }
	const int get_n_landmark() const { return n_landmark; }
	
	double *get_mutable_shape_coef() { return shape_coef; }
	double *get_mutable_tex_coef() { return tex_coef; }
	double *get_mutable_expr_coef() { return expr_coef; }

	double *get_mutable_extrinsic_params() { return extrinsic_params; }
	double *get_mutable_intrinsic_params() { return intrinsic_params; }
	const double *get_extrinsic_params() const { return extrinsic_params; }
	const double *get_intrinsic_params() const { return intrinsic_params; }
	const dlib::matrix<double> get_R() const { return R; }
	const dlib::matrix<double> get_T() const { return T; }

	const double get_fx() const { return intrinsic_params[0]; }
	const double get_fy() const { return intrinsic_params[1]; }
	const double get_cx() const { return intrinsic_params[2]; }
	const double get_cy() const { return intrinsic_params[3]; }

	const double get_yaw() const { return extrinsic_params[0]; }
	const double get_pitch() const { return extrinsic_params[1]; }
	const double get_roll() const { return extrinsic_params[2]; }
	const double get_tx() const { return extrinsic_params[3]; }
	const double get_ty() const { return extrinsic_params[4]; }
	const double get_tz() const { return extrinsic_params[5]; }

	void set_yaw(double yaw) { extrinsic_params[0] = yaw; }
	void set_pitch(double pitch) { extrinsic_params[1] = pitch; }
	void set_roll(double roll) { extrinsic_params[2] = roll; }
	void set_rotation(double yaw, double pitch, double roll) 
	{
		set_yaw(yaw); set_pitch(pitch); set_roll(roll);
	}
	void set_R(const dlib::matrix<double> &R_) { R = R_; }
	void set_R(const cv::Mat &R_) 
	{ 
		R(0, 0) = R_.at<double>(0, 0); R(0, 1) = R_.at<double>(0, 1); R(0, 2) = R_.at<double>(0, 2);
		R(1, 0) = R_.at<double>(1, 0); R(1, 1) = R_.at<double>(1, 1); R(1, 2) = R_.at<double>(1, 2);
		R(2, 0) = R_.at<double>(2, 0); R(2, 1) = R_.at<double>(2, 1); R(2, 2) = R_.at<double>(2, 2);
	}
	void set_R(CvMat *R_)
	{
		R(0, 0) = cvmGet(R_, 0, 0); R(0, 1) = cvmGet(R_, 0, 1); R(0, 2) = cvmGet(R_, 0, 2);
		R(1, 0) = cvmGet(R_, 1, 0); R(1, 1) = cvmGet(R_, 1, 1); R(1, 2) = cvmGet(R_, 1, 2);
		R(2, 0) = cvmGet(R_, 2, 0); R(2, 1) = cvmGet(R_, 2, 1); R(2, 2) = cvmGet(R_, 2, 2);		
	}
	void set_T(const dlib::matrix<double> &_T) { T = _T; }
	void set_T(const cv::Mat &_T)
	{
		T(0, 0) = _T.at<double>(0, 0);
		T(1, 0) = _T.at<double>(1, 0);
		T(2, 0) = _T.at<double>(2, 0);
 	}
	void set_T(CvMat *T_)
	{
		T(0, 0) = cvmGet(T_, 0, 0);
		T(1, 0) = cvmGet(T_, 1, 0);
		T(2, 0) = cvmGet(T_, 2, 0);		
	}

	void set_tx(double tx) { extrinsic_params[3] = tx; }
	void set_ty(double ty) { extrinsic_params[4] = ty; }
	void set_tz(double tz) { extrinsic_params[5] = tz; }

	const dlib::matrix<double> &get_current_shape() const { return current_shape; }
	const dlib::matrix<double> &get_current_tex() const { return current_tex; }
	const dlib::matrix<double> &get_current_expr() const { return current_expr; }
	const dlib::matrix<double> &get_current_blendshape() const { return current_blendshape; }
	const dlib::matrix<double> &get_fp_current_blendshape() const { return fp_current_blendshape; }
	dlib::matrix<double> get_fp_current_blendshape_transformed() const;
	const dlib::matrix<double> &get_tl() const { return tl; }

#ifndef BFM_SHUT_UP
	void print_fp_shape_mu() const { bfm_out << "landmark - shape average:\n"  << fp_shape_mu; }
	void print_fp_shape_pc() const { bfm_out << "landmark - shape pc:\n"	   << fp_shape_pc; }
	void print_shape_ev() const { bfm_out << "shape variance:\n " << shape_ev; }
	void print_extrinsic_params() const;
	void print_intrinsic_params() const;
	void print_shape_coef() const 
	{ 
		bfm_out << "shape coef:\n";
		for(int i=0; i<n_id_pc; i++) bfm_out << shape_coef[i] << "\n";
	}
	void print_expr_coef() const 
	{ 
		bfm_out << "expression coef:\n";
		for(int i=0; i<n_expr_pc; i++) bfm_out << expr_coef[i] << "\n";
	}
	inline void print_R() const { bfm_out << "R: \n" << R; }
	inline void print_T() const { bfm_out << "T: \n" << T; }
#endif

	void clear_ext_params();
	void write_ext_params_to_file(std::ofstream &f) const
	{
		f << extrinsic_params[0] << " ";
		f << extrinsic_params[1] << " ";
		f << extrinsic_params[2] << " ";
		f << extrinsic_params[3] << " ";
		f << extrinsic_params[4] << " ";
		f << extrinsic_params[5] << "\n";
	}
	void write_fp_to_file(std::ofstream &f) const 
	{
		for(int i=0; i<n_landmark; i++)
			f << fp_current_blendshape(i*3) << " " 
			  << fp_current_blendshape(i*3+1) << " "
			  << fp_current_blendshape(i*3+2);
	}

private:
	bool read_params_from_file(const std::string &filename);
	void init_params();
	bool load_data();
	void extract_landmark();
	template<typename T>
	dlib::matrix<T> coef2object(const T *const &coef, const dlib::matrix<double> &mu,
							    const dlib::matrix<double> &pc, const dlib::matrix<double> &ev, int len) const 
	{ 
		dlib::matrix<T> coef_(len, 1);
		for(int i=0; i<len; i++)
			coef_(i) = coef[i];

		dlib::matrix<T> mu_ = dlib::matrix_cast<T>(mu);
		dlib::matrix<T> pc_ = dlib::matrix_cast<T>(pc);
		dlib::matrix<T> ev_ = dlib::matrix_cast<T>(ev);
		return mu_ + pc_ * pointwise_multiply(coef_, ev_);
	}


	bool use_landmark;

	std::string bfm_h5_path;
	std::string landmark_idx_path;

	std::string shape_mu_h5_path;
	std::string shape_ev_h5_path;
	std::string shape_pc_h5_path;

	std::string tex_mu_h5_path;
	std::string tex_ev_h5_path;
	std::string tex_pc_h5_path;

	std::string expr_mu_h5_path;
	std::string expr_ev_h5_path;
	std::string expr_pc_h5_path;

	std::string tl_h5_path;

	int n_vertice;
	int n_face;
	int n_id_pc;
	int n_expr_pc;
	int n_landmark;

	/* ZYX - euler angle */
	/* yaw:   rotate around z axis */
	/* pitch: rotate around y axis */
    /* roll:  rotate around x axis */
	dlib::matrix<double, 3, 3> R;
	dlib::matrix<double, 3, 1> T;
	double extrinsic_params[6] = { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };	/* yaw pitch roll tx ty tz */
	double intrinsic_params[4] = { 0.f };	/* fx fy cx cy */

	double *shape_coef;
	dlib::matrix<double> shape_mu;
	dlib::matrix<double> shape_ev;
	dlib::matrix<double> shape_pc;

	double *tex_coef;
	dlib::matrix<double> tex_mu;
	dlib::matrix<double> tex_ev;
	dlib::matrix<double> tex_pc;

	double *expr_coef;
	dlib::matrix<double> expr_mu;
	dlib::matrix<double> expr_ev;
	dlib::matrix<double> expr_pc;

	dlib::matrix<double> fp_shape_mu;
	dlib::matrix<double> fp_shape_pc;
	dlib::matrix<double> fp_expr_mu;
	dlib::matrix<double> fp_expr_pc;

	dlib::matrix<double> tl;	/* triangle list */

	dlib::matrix<double> current_shape;
	dlib::matrix<double> current_tex;
	dlib::matrix<double> current_expr;
	dlib::matrix<double> current_blendshape;

	dlib::matrix<double> fp_current_shape;
	dlib::matrix<double> fp_current_expr;
	dlib::matrix<double> fp_current_blendshape;

	std::vector<int> landmark_idx;
};