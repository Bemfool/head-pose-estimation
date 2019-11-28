#include "data.h"

/* data collection */
dlib::matrix<double> shape_coef(N_ID_PC, 1);
dlib::matrix<double> shape_mu(N_VERTICE * 3, 1);
dlib::matrix<double> shape_ev(N_ID_PC, 1);
dlib::matrix<double> shape_pc(N_VERTICE * 3, N_ID_PC);

dlib::matrix<double> tex_coef(N_ID_PC, 1);
dlib::matrix<double> tex_mu(N_VERTICE * 3, 1);
dlib::matrix<double> tex_ev(N_ID_PC, 1);
dlib::matrix<double> tex_pc(N_VERTICE * 3, N_ID_PC);

dlib::matrix<double> expr_coef(N_EXPR_PC, 1);
dlib::matrix<double> expr_mu(N_VERTICE * 3, 1);
dlib::matrix<double> expr_ev(N_EXPR_PC, 1);
dlib::matrix<double> expr_pc(N_VERTICE * 3, N_EXPR_PC);

std::vector<point3d> tl(N_FACE);	/* triangle list */

dlib::matrix<double> current_shape(N_VERTICE * 3, 1);
dlib::matrix<double> current_tex(N_VERTICE * 3, 1);
dlib::matrix<double> current_expr(N_VERTICE * 3, 1);
dlib::matrix<double> current_blendshape(N_VERTICE * 3, 1);

std::vector<double> landmark_idx(LANDMARK_NUM);

/* head pose parameters */
double yaw, pitch, roll;	/* rotation */
double tx, ty, tz;			/* translation */


int load() {
	std::cout << "data list: " << std::endl;
	ifstream in;
	int res = 0;
	float *shape_mu_raw  = new float[N_VERTICE * 3];
	float *shape_ev_raw  = new float[N_ID_PC];
	float *shape_pc_raw  = new float[N_VERTICE * 3 * N_ID_PC];
	float *tex_mu_raw  = new float[N_VERTICE * 3];
	float *tex_ev_raw  = new float[N_ID_PC];
	float *tex_pc_raw  = new float[N_VERTICE * 3 * N_ID_PC];
	float *expr_mu_raw  = new float[N_VERTICE * 3];
	float *expr_ev_raw  = new float[N_EXPR_PC];
	float *expr_pc_raw  = new float[N_VERTICE * 3 * N_EXPR_PC];
	unsigned int *tl_raw = new unsigned int[N_FACE * 3];

	std::cout << "trying to open " << bfm_h5_path << std::endl;
	H5File file(bfm_h5_path, H5F_ACC_RDONLY);
	load_hdf5_model(shape_mu, "/shape/model/mean",             PredType::NATIVE_FLOAT);
	load_hdf5_model(shape_ev, "/shape/model/pcaVariance",      PredType::NATIVE_FLOAT);
	load_hdf5_model(shape_pc, "/shape/model/pcaBasis",         PredType::NATIVE_FLOAT);

	load_hdf5_model(tex_mu,   "/color/model/mean",             PredType::NATIVE_FLOAT);
	load_hdf5_model(tex_ev,   "/color/model/pcaVariance",      PredType::NATIVE_FLOAT);
	load_hdf5_model(tex_pc,   "/color/model/pcaBasis",    	   PredType::NATIVE_FLOAT);	
	
	load_hdf5_model(expr_mu,  "/expression/model/mean",        PredType::NATIVE_FLOAT);	
	load_hdf5_model(expr_ev,  "/expression/model/pcaVariance", PredType::NATIVE_FLOAT);	
	load_hdf5_model(expr_pc,  "/expression/model/pcaBasis",    PredType::NATIVE_FLOAT);	
	
	load_hdf5_model(tl,       "/color/representer/cells",      PredType::NATIVE_UINT32);
	file.close();

	shape_mu = shape_mu * 1000.0;
	tex_mu = tex_mu * 255.0;

	return (res < 0) ? FAIL : SUCCESS;
}

void raw2matrix(dlib::matrix<double> &m, float *raw) {
	std::cout << "check: " << m.nr() << " " << m.nc() << std::endl;
	for (int i = 0; i < m.nr(); i++)
		for(int j = 0; j < m.nc(); j++)
			m(i, j) = raw[i * m.nc() + j];
}

void raw2matrix(std::vector<point3d> &vec, unsigned int *raw) {
	for (int i = 0; i < N_FACE; i++) {
		vec[i].x() = raw[i];
		vec[i].y() = raw[i + N_FACE];
		vec[i].z() = raw[i + N_FACE * 2];
	}
}


void data_check() {
	std::cout << "check data" << std::endl;
	std::cout << "	(1) shape mu: " << std::endl;
	std::cout << "		Yours:   " << shape_mu(0, 0) << std::endl;
	std::cout << "		Correct: -57239 42966 80410\n" << std::endl;
	std::cout << "	(2) shape ev: " << std::endl;
	std::cout << "		Yours:   " << shape_ev(0, 0) << " " << shape_ev(1, 0) << std::endl;
	std::cout << "		Correct: 884340 555880\n" << std::endl;
	std::cout << "	(3) shape pc: " << std::endl;
	std::cout << "		Yours:   " << shape_pc(0, 0) << std::endl;
	std::cout << "		Correct: -0.0024\n" << std::endl;
	std::cout << "	(4) texture mu: " << std::endl;
	std::cout << "		Yours:   " << tex_mu(0, 0) << endl;
	std::cout << "		Correct: 182.8750 135.0400 107.1400\n" << endl;
	std::cout << "	(5) texture ev: " << std::endl;
	std::cout << "		Yours:   " << tex_ev(0) << " " << tex_ev(1) << std::endl;
	std::cout << "		Correct: 4103.2 2024.1\n" << std::endl;
	std::cout << "	(6) texture pc: " << std::endl;
	std::cout << "		Yours:   " << tex_pc(0, 0) << std::endl;
	std::cout << "		Correct: -0.0028\n" << std::endl;
	std::cout << "	(7) expression mu: " << std::endl;
	std::cout << "		Yours:   " << expr_mu(0, 0) << endl;
	std::cout << "		Correct: 182.8750 135.0400 107.1400\n" << endl;
	std::cout << "	(8) expression ev: " << std::endl;
	std::cout << "		Yours:   " << expr_ev(0) << " " << expr_ev(1) << std::endl;
	std::cout << "		Correct: 4103.2 2024.1\n" << std::endl;
	std::cout << "	(9) expression pc: " << std::endl;
	std::cout << "		Yours:   " << expr_pc(0, 0) << std::endl;
	std::cout << "		Correct: -0.0028\n" << std::endl;
	std::cout << "	(10) triangle list: " << std::endl;
	std::cout << "		Yours:   " << tl.at(0).x() << " " << tl.at(1).y() << std::endl;
	std::cout << "		Correct: -0.0028\n" << std::endl;
	
}

void set_shape_pc_basis(const double *x) {
	for(int i=0; i<shape_coef.nr(); i++)
		shape_coef(i, 0) = x[i];
}

void set_expr_pc_basis(const double *x) {
	for(int i=0; i<expr_coef.nr(); i++)
		expr_coef(i, 0) = x[i];
}

void set_head_pose_parameters(const double *x) {
	yaw   = x[0];
	pitch = x[1];
	roll  = x[2];
	tx    = x[3];
	ty    = x[4];
	tz    = x[5];
}