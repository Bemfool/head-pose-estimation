﻿#include "bfm.h"

bfm::bfm(const std::string filename) {
	init(filename);
}


void bfm::init(const std::string filename) {
	if (!read_parm_from_file(filename))
		return;
	init_parm();
	load_data();
	if(use_landmark) 
		extract_landmark();
	generate_average_face();
	generate_fp_face();
}


void bfm::data_check() const {
	bfm_out << "check data\n";
	bfm_out << "	(1) shape mu: \n";
	bfm_out << "		Yours:   " << shape_mu(0, 0) << "\n";
	bfm_out << "		Correct: -57239 42966 80410\n\n";
	bfm_out << "	(2) shape ev: \n";
	bfm_out << "		Yours:   " << shape_ev(0, 0) << " " << shape_ev(1, 0) << "\n";
	bfm_out << "		Correct: 884340 555880\n\n";
	bfm_out << "	(3) shape pc: \n";
	bfm_out << "		Yours:   " << shape_pc(0, 0) << "\n";
	bfm_out << "		Correct: -0.0024\n\n";
	bfm_out << "	(4) texture mu: \n";
	bfm_out << "		Yours:   " << tex_mu(0, 0) << endl;
	bfm_out << "		Correct: 182.8750 135.0400 107.1400\n" << endl;
	bfm_out << "	(5) texture ev: \n";
	bfm_out << "		Yours:   " << tex_ev(0) << " " << tex_ev(1) << "\n";
	bfm_out << "		Correct: 4103.2 2024.1\n\n";
	bfm_out << "	(6) texture pc: \n";
	bfm_out << "		Yours:   " << tex_pc(0, 0) << "\n";
	bfm_out << "		Correct: -0.0028\n\n";
	bfm_out << "	(7) expression mu: \n";
	bfm_out << "		Yours:   " << expr_mu(0, 0) << endl;
	bfm_out << "		Correct: 182.8750 135.0400 107.1400\n" << endl;
	bfm_out << "	(8) expression ev: \n";
	bfm_out << "		Yours:   " << expr_ev(0) << " " << expr_ev(1) << "\n";
	bfm_out << "		Correct: 4103.2 2024.1\n\n";
	bfm_out << "	(9) expression pc: \n";
	bfm_out << "		Yours:   " << expr_pc(0, 0) << "\n";
	bfm_out << "		Correct: -0.0028\n\n";
	bfm_out << "	(10) triangle list: \n";
	bfm_out << "		Yours:   " << tl(0) << " " << tl(1) << "\n";
	bfm_out << "		Correct: -0.0028\n\n";
}


bool bfm::read_parm_from_file(const std::string &filename) {
	ifstream in(filename, std::ios::in);
	if (!in) {
		bfm_out << "[ERROR] Can't open " << filename.c_str() << ".\n";
		return false;
	}
	in >> bfm_h5_path;
	in >> n_vertice >> n_face >> n_id_pc >> n_expr_pc;
	in >> n_landmark >> landmark_idx_path;
	use_landmark = (n_landmark == -1) ? false : true;
	in >> intrinsic_parm[0] >> intrinsic_parm[1] >> intrinsic_parm[2] >> intrinsic_parm[3];
	in >> shape_mu_h5_path >> shape_ev_h5_path >> shape_pc_h5_path;
	in >> tex_mu_h5_path >> tex_ev_h5_path >> tex_pc_h5_path;
	in >> expr_mu_h5_path >> expr_ev_h5_path >> expr_pc_h5_path;
	in >> tl_h5_path;

	in.close();
	return true;
}


void bfm::init_parm() {
	shape_coef = new double[n_id_pc];
	fill(shape_coef, shape_coef + n_id_pc, 0.f);
	shape_mu.set_size(n_vertice * 3, 1);
	shape_ev.set_size(n_id_pc, 1);
	shape_pc.set_size(n_vertice * 3, n_id_pc);

	tex_coef = new double[n_id_pc];
	fill(tex_coef, tex_coef + n_id_pc, 0.f);
	tex_mu.set_size(n_vertice * 3, 1);
	tex_ev.set_size(n_id_pc, 1);
	tex_pc.set_size(n_vertice * 3, n_id_pc);

	expr_coef = new double[n_expr_pc];
	fill(expr_coef, expr_coef + n_expr_pc, 0.f);
	expr_mu.set_size(n_vertice * 3, 1);
	expr_ev.set_size(n_expr_pc, 1);
	expr_pc.set_size(n_vertice * 3, n_expr_pc);

	tl.set_size(n_face * 3, 1);

	current_shape.set_size(n_vertice * 3, 1);
	current_tex.set_size(n_vertice * 3, 1);
	current_expr.set_size(n_vertice * 3, 1);
	current_blendshape.set_size(n_vertice * 3, 1);

	if (use_landmark) {
		landmark_idx.resize(n_landmark);
		fp_shape_mu.set_size(n_landmark * 3, 1);
		fp_shape_pc.set_size(n_landmark * 3, n_id_pc);
		fp_expr_mu.set_size(n_landmark * 3, 1);
		fp_expr_pc.set_size(n_landmark * 3, n_expr_pc);
	}
}


bool bfm::load_data() {
	float *shape_mu_raw = new float[n_vertice * 3];
	float *shape_ev_raw = new float[n_id_pc];
	float *shape_pc_raw = new float[n_vertice * 3 * n_id_pc];
	float *tex_mu_raw = new float[n_vertice * 3];
	float *tex_ev_raw = new float[n_id_pc];
	float *tex_pc_raw = new float[n_vertice * 3 * n_id_pc];
	float *expr_mu_raw = new float[n_vertice * 3];
	float *expr_ev_raw = new float[n_expr_pc];
	float *expr_pc_raw = new float[n_vertice * 3 * n_expr_pc];
	unsigned int *tl_raw = new unsigned int[n_face * 3];

	H5File file(bfm_h5_path, H5F_ACC_RDONLY);
	load_hdf5_model(shape_mu, shape_mu_h5_path, PredType::NATIVE_FLOAT);
	load_hdf5_model(shape_ev, shape_ev_h5_path, PredType::NATIVE_FLOAT);
	load_hdf5_model(shape_pc, shape_pc_h5_path, PredType::NATIVE_FLOAT);

	load_hdf5_model(tex_mu, tex_mu_h5_path, PredType::NATIVE_FLOAT);
	load_hdf5_model(tex_ev, tex_ev_h5_path, PredType::NATIVE_FLOAT);
	load_hdf5_model(tex_pc, tex_pc_h5_path, PredType::NATIVE_FLOAT);

	load_hdf5_model(expr_mu, expr_mu_h5_path, PredType::NATIVE_FLOAT);
	load_hdf5_model(expr_ev, expr_ev_h5_path, PredType::NATIVE_FLOAT);
	load_hdf5_model(expr_pc, expr_pc_h5_path, PredType::NATIVE_FLOAT);

	load_hdf5_model(tl, tl_h5_path, PredType::NATIVE_UINT32);
	file.close();

	shape_mu = shape_mu * 1000.0;

	ifstream in(landmark_idx_path, std::ios::in);
	if (!in) {
		bfm_out << "[ERROR] Can't open " << landmark_idx_path.c_str() << ".\n";
		return false;
	}
	for (int i = 0; i < n_landmark; i++) {
		int tmp_idx;
		in >> tmp_idx;
		landmark_idx[i] = tmp_idx - 1;
	}
	return true;
}

void bfm::extract_landmark() {
	for(int i=0; i<n_landmark; i++) {
		int idx = landmark_idx[i];
		fp_shape_mu(i*3) = shape_mu(idx*3);
		fp_shape_mu(i*3+1) = shape_mu(idx*3+1);
		fp_shape_mu(i*3+2) = shape_mu(idx*3+2);
		fp_expr_mu(i*3) = expr_mu(idx*3);
		fp_expr_mu(i*3+1) = expr_mu(idx*3+1);
		fp_expr_mu(i*3+2) = expr_mu(idx*3+2);
		for(int j=0; j<n_id_pc; j++) {
			fp_shape_pc(i*3, j) = shape_pc(idx*3, j);
			fp_shape_pc(i*3+1, j) = shape_pc(idx*3+1, j);
			fp_shape_pc(i*3+2, j) = shape_pc(idx*3+2, j);	
			// std::cout << "loading fp " <<  fp_shape_pc(i*3) << " " << fp_shape_pc(i*3+1) << " " << fp_shape_pc(i*3+2) << std::endl;		
		}
		for(int j=0; j<n_expr_pc; j++) {
			fp_expr_pc(i*3, j) = expr_pc(idx*3, j);
			fp_expr_pc(i*3+1, j) = expr_pc(idx*3+1, j);
			fp_expr_pc(i*3+2, j) = expr_pc(idx*3+2, j);
		}
	}
}

void bfm::print_external_parm() const {
	bfm_out << "yaw: "   << external_parm[0] << "\n";
	bfm_out << "pitch: " << external_parm[1] << "\n";
	bfm_out << "roll: "  << external_parm[2] << "\n";
	bfm_out << "tx: "    << external_parm[3] << "\n";
	bfm_out << "ty: "    << external_parm[4] << "\n";
	bfm_out << "tz: "    << external_parm[5] << "\n";
}


void bfm::print_intrinsic_parm() const {
	bfm_out << "fx: " << intrinsic_parm[0] << "\n";
	bfm_out << "fy: " << intrinsic_parm[1] << "\n";
	bfm_out << "cx: " << intrinsic_parm[2] << "\n";
	bfm_out << "cy: " << intrinsic_parm[3] << "\n";
}


void bfm::generate_random_face(double scale) {
	shape_coef = randn(n_id_pc, scale);
	tex_coef = randn(n_id_pc, scale);
	expr_coef = randn(n_expr_pc, scale);
	generate_face();
}


void bfm::generate_random_face(double shape_scale, double tex_scale, double expr_scale) {
	shape_coef = randn(n_id_pc, shape_scale);
	tex_coef = randn(n_id_pc, tex_scale);
	expr_coef = randn(n_expr_pc, expr_scale);
	generate_face();
}


void bfm::generate_face() {
	current_shape = coef2object(shape_coef, shape_mu, shape_pc, shape_ev, n_id_pc);
	current_tex = coef2object(tex_coef, tex_mu, tex_pc, tex_ev, n_id_pc);
	current_expr = coef2object(expr_coef, expr_mu, expr_pc, expr_ev, n_expr_pc);
	current_blendshape = current_shape + current_expr;
}


void bfm::generate_fp_face() {
	fp_current_shape = coef2object(shape_coef, fp_shape_mu, fp_shape_pc, shape_ev, n_id_pc);
	// print_shape_ev();
	// print_fp_shape_pc();
	// print_shape_coef();

	fp_current_expr = coef2object(expr_coef, fp_expr_mu, fp_expr_pc, expr_ev, n_expr_pc);
	fp_current_blendshape = fp_current_shape + fp_current_expr;
}


void bfm::ply_write(std::string fn, long mode) const {
	std::ofstream out;
	/* Note: In Linux Cpp, we should use std::ios::bfm_out as flag, which is not necessary in Windows */
	out.open(fn, std::ios::out | std::ios::binary);
	if (!out) {
		bfm_out << "Creation of " << fn.c_str() << " failed.\n";
		return;
	}
	out << "ply\n";
	out << "format binary_little_endian 1.0\n";
	out << "comment Made from the 3D Morphable Face Model of the Univeristy of Basel, Switzerland.\n";
	out << "element vertex " << n_vertice << "\n";
	out << "property float x\n";
	out << "property float y\n";
	out << "property float z\n";
	out << "property uchar red\n";
	out << "property uchar green\n";
	out << "property uchar blue\n";
	out << "element face " << n_face << "\n";
	out << "property list uchar int vertex_indices\n";
	out << "end_header\n";

	int cnt = 0;
	for (int i = 0; i < n_vertice; i++) {
		float x, y, z;
		if(mode & NO_EXPR) {
			x = float(current_shape(i * 3));
			y = float(current_shape(i * 3 + 1));
			z = float(current_shape(i * 3 + 2));
		} else {
			x = float(current_blendshape(i * 3));
			y = float(current_blendshape(i * 3 + 1));
			z = float(current_blendshape(i * 3 + 2));
		}

		if(mode & CAMERA_COORD) {
			transform(external_parm, x, y, z);
			y = -y; z = -z;
		}

		unsigned char r, g, b;
		if ((mode & PICK_FP) && find(landmark_idx.begin(), landmark_idx.end(), i) != landmark_idx.end()) {
			r = 255;
			g = 0;
			b = 0;
			cnt++;
		} else {
			r = current_tex(i * 3);
			g = current_tex(i * 3 + 1);
			b = current_tex(i * 3 + 2);
		}

		out.write((char *)&x, sizeof(x));
		out.write((char *)&y, sizeof(y));
		out.write((char *)&z, sizeof(z));
		out.write((char *)&r, sizeof(r));
		out.write((char *)&g, sizeof(g));
		out.write((char *)&b, sizeof(b));
	}
	if ((mode & PICK_FP) && cnt != n_landmark) {
		bfm_out << "[ERROR] Pick too less landmarks.\n";
		bfm_out << "Number of picked points is " << cnt << ".\n";
	}

	unsigned char N_VER_PER_FACE = 3;
	for (int i = 0; i < n_face; i++) {
		out.write((char *)&N_VER_PER_FACE, sizeof(N_VER_PER_FACE));
		int x = tl(i * 3) - 1;
		int y = tl(i * 3 + 1) - 1;
		int z = tl(i * 3 + 2) - 1;
		out.write((char *)&y, sizeof(y));
		out.write((char *)&x, sizeof(x));
		out.write((char *)&z, sizeof(z));
	}
	out.close();
}

void bfm::ply_write_fp(std::string fn) const {
	std::ofstream out;
	/* Note: In Linux Cpp, we should use std::ios::bfm_out as flag, which is not necessary in Windows */
	out.open(fn, std::ios::out | std::ios::binary);
	if (!out) {
		bfm_out << "Creation of " << fn.c_str() << " failed.\n";
		return;
	}
	out << "ply\n";
	out << "format binary_little_endian 1.0\n";
	out << "comment Made from the 3D Morphable Face Model of the Univeristy of Basel, Switzerland.\n";
	out << "element vertex " << n_landmark << "\n";
	out << "property float x\n";
	out << "property float y\n";
	out << "property float z\n";
	out << "end_header\n";

	int cnt = 0;
	for (int i = 0; i < n_landmark; i++) {
		float x, y, z;
		x = float(fp_current_blendshape(i * 3));
		y = float(fp_current_blendshape(i * 3 + 1));
		z = float(fp_current_blendshape(i * 3 + 2));
		out.write((char *)&x, sizeof(x));
		out.write((char *)&y, sizeof(y));
		out.write((char *)&z, sizeof(z));
	}

	out.close();	
}