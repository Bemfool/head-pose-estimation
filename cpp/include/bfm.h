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
	dlib::matrix<T> generate_fp_face(const T * const shape_coef_,const T * const expr_coef_) const {
		dlib::matrix<T> fp_current_shape_ = coef2object(shape_coef_, fp_shape_mu, fp_shape_pc, shape_ev, n_id_pc);
		dlib::matrix<T> fp_current_expr_ = coef2object(expr_coef_, fp_expr_mu, fp_expr_pc, expr_ev, n_expr_pc);
		dlib::matrix<T> fp_current_blendshape_ = fp_current_shape_ + fp_current_expr_;	
		return fp_current_blendshape_;
	}
	template<typename T>
	dlib::matrix<T> generate_fp_face_by_shape(const T * const shape_coef_) const {
		dlib::matrix<T> fp_current_shape_ = coef2object(shape_coef_, fp_shape_mu, fp_shape_pc, shape_ev, n_id_pc);
		// std::cout << "fp_current_shape: " << fp_current_shape_(0, 0);
		// dlib::matrix<T> fp_current_expr_ = dlib::matrix_cast<T>(coef2object(expr_coef, fp_expr_mu, fp_expr_pc, expr_ev));
		// std::cout << "fp_current_expr: " << fp_current_expr(0, 0);
		dlib::matrix<T> fp_current_expr_ = dlib::matrix_cast<T>(fp_current_expr);
		// std::cout << "fp_current_expr_: " << fp_current_expr_(0, 0);
		dlib::matrix<T> fp_current_blendshape_ = fp_current_shape_ + fp_current_expr_;	
		return fp_current_blendshape_;		
	}
	template<typename T>
	dlib::matrix<T> generate_fp_face_by_expr(const T * const expr_coef_) const {
		dlib::matrix<T> fp_current_shape_ = dlib::matrix_cast<T>(coef2object(shape_coef, fp_shape_mu, fp_shape_pc, shape_ev, n_id_pc));
		dlib::matrix<T> fp_current_expr_ = coef2object(expr_coef_, fp_expr_mu, fp_expr_pc, expr_ev, n_expr_pc);
		dlib::matrix<T> fp_current_blendshape_ = fp_current_shape_ + fp_current_expr_;	
		return fp_current_blendshape_;		
	}

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

	double *get_mutable_external_parm() { return external_parm; }
	double *get_mutable_intrinsic_parm() { return intrinsic_parm; }
	const double *get_external_parm() const { return external_parm; }
	const double *get_intrinsic_parm() const { return intrinsic_parm; }

	const double get_fx() const { return intrinsic_parm[0]; }
	const double get_fy() const { return intrinsic_parm[1]; }
	const double get_cx() const { return intrinsic_parm[2]; }
	const double get_cy() const { return intrinsic_parm[3]; }

	const double get_yaw() const { return external_parm[0]; }
	const double get_pitch() const { return external_parm[1]; }
	const double get_roll() const { return external_parm[2]; }
	const double get_tx() const { return external_parm[3]; }
	const double get_ty() const { return external_parm[4]; }
	const double get_tz() const { return external_parm[5]; }

	void set_yaw(double yaw) { external_parm[0] = yaw; }
	void set_pitch(double pitch) { external_parm[1] = pitch; }
	void set_roll(double roll) { external_parm[2] = roll; }
	void set_rotation(double yaw, double pitch, double roll) {
		set_yaw(yaw); set_pitch(pitch); set_roll(roll);
	}
	void set_tx(double tx) { external_parm[3] = tx; }
	void set_ty(double ty) { external_parm[4] = ty; }
	void set_tz(double tz) { external_parm[5] = tz; }

	const dlib::matrix<double> &get_current_shape() const { return current_shape; }
	const dlib::matrix<double> &get_current_tex() const { return current_tex; }
	const dlib::matrix<double> &get_current_expr() const { return current_expr; }
	const dlib::matrix<double> &get_current_blendshape() const { return current_blendshape; }
	const dlib::matrix<double> &get_fp_current_blendshape() const { return fp_current_blendshape; }
	const dlib::matrix<double> &get_tl() const { return tl; }

	void print_fp_shape_mu() const { bfm_out << "landmark - shape average:\n"  << fp_shape_mu; }
	void print_fp_shape_pc() const { bfm_out << "landmark - shape pc:\n"	   << fp_shape_pc; }
	void print_shape_ev() const { bfm_out << "shape variance:\n " << shape_ev; }
	void print_external_parm() const;
	void print_intrinsic_parm() const;
	void print_shape_coef() const { 
		bfm_out << "shape coef:\n";
		for(int i=0; i<n_id_pc; i++) bfm_out << shape_coef[i] << "\n";
	}
	void print_expr_coef() const { 
		bfm_out << "expression coef:\n";
		for(int i=0; i<n_expr_pc; i++) bfm_out << expr_coef[i] << "\n";
	}

private:
	bool read_parm_from_file(const std::string &filename);
	void init_parm();
	bool load_data();
	void extract_landmark();
	template<typename T>
	dlib::matrix<T> coef2object(const T *const &coef, const dlib::matrix<double> &mu,
							    const dlib::matrix<double> &pc, const dlib::matrix<double> &ev, int len) const { 
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

	/* euler angle */
	/* yaw:   rotate around z axis */
	/* pitch: rotate around y axis */
    /* roll:  rotate around x axis */
	double external_parm[6] = { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };	/* yaw pitch roll tx ty tz */
	double intrinsic_parm[4] = { 0.f };	/* fx fy cx cy */

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