#include "data.h"

/* data collection */
std::vector<double> alpha;	/* coeficients of shape */
std::vector<double> beta;   /* coeficients of texture */

std::vector<point3f> shape_mu(N_VERTICE);			/* mean of shape */
std::vector<double> shape_ev(N_PC);					/* pc variance of shape */
std::vector<std::vector<point3f>> shape_pc(N_PC);   /* pc basic of shape */
std::vector<point3f> tex_mu(N_VERTICE);				/* mean of texture */
std::vector<double> tex_ev(N_PC);					/* pc vairance of texture */
std::vector<std::vector<point3f>> tex_pc(N_PC);		/* pc basic of texture */

std::vector<point3d> tl(N_FACE_HDF5);	/* triangle list */

std::vector<point3f> shape;	/* current shape */
std::vector<point3f> tex;	/* current texture */


int load() {
	std::cout << "data list: " << std::endl;
	ifstream in;
	int res = 0;
	float *shape_mu_raw  = new float[N_VERTICE * 3];
	float *shape_ev_raw  = new float[N_PC];
	float *shape_pc_raw  = new float[N_VERTICE * 3 * N_PC];
	float *tex_mu_raw    = new float[N_VERTICE * 3];
	float *tex_ev_raw    = new float[N_PC];
	float *tex_pc_raw    = new float[N_VERTICE * N_PC * 3];
	unsigned int *tl_raw = new unsigned int[N_FACE_HDF5 * 3];

	std::cout << "trying to open " << bfm_h5_path << std::endl;
	H5File file(bfm_h5_path, H5F_ACC_RDONLY);
	load_hdf5_model(shape_mu, "/shape/model/mean",        PredType::NATIVE_FLOAT);
	load_hdf5_model(shape_ev, "/shape/model/pcaVariance", PredType::NATIVE_FLOAT);
	load_hdf5_model(shape_pc, "/shape/model/pcaBasis",    PredType::NATIVE_FLOAT);
	load_hdf5_model(tex_mu,   "/color/model/mean",        PredType::NATIVE_FLOAT);
	load_hdf5_model(tex_ev,   "/color/model/pcaVariance", PredType::NATIVE_FLOAT);
	load_hdf5_model(tex_pc,   "/color/model/pcaBasis",    PredType::NATIVE_FLOAT);
	load_hdf5_model(tl,       "/color/representer/cells", PredType::NATIVE_UINT32);
	file.close();

	for (int i = 0; i < N_VERTICE; i++) {
		shape_mu[i] = shape_mu[i] * 1000.0;
		tex_mu[i] = tex_mu[i] * 255.0;
	}

	return (res < 0) ? FAIL : SUCCESS;
}

void raw2vector(std::vector<double> &vec, float *raw) {
	for (int i = 0; i < N_PC; i++)
		vec[i] = raw[i];
}

void raw2vector(std::vector<point3f> &vec, float *raw) {
	for (int i = 0; i < N_VERTICE; i++) {
		vec[i].x() = raw[i * 3];
		vec[i].y() = raw[i * 3 + 1];
		vec[i].z() = raw[i * 3 + 2];
	}
}

void raw2vector(std::vector<point3d> &vec, unsigned int *raw) {
	for (int i = 0; i < N_FACE_HDF5; i++) {
		vec[i].x() = raw[i];
		vec[i].y() = raw[i + N_FACE_HDF5];
		vec[i].z() = raw[i + N_FACE_HDF5 * 2];
	}
}

void raw2vector(std::vector<std::vector<point3f>> &vec, float *raw) {
	for (int i = 0; i < N_VERTICE; i++) {
		for (int j = 0; j < N_PC; j++) {
				vec[j][i].x() = raw[j + (i * 3) * N_PC];
				vec[j][i].y() = raw[j + (i * 3 + 1) * N_PC];
				vec[j][i].z() = raw[j + (i * 3 + 2) * N_PC];
		}
	}
}


void data_check() {
	cout << "check data" << endl;
	cout << "(1) shape mu: " << endl;
	cout << "Yours:   " << shape_mu[0] << endl;
	cout << "Correct: -57239 42966 80410" << endl;
	cout << endl;
	cout << "(2) shape ev: " << endl;
	cout << "Yours:   " << shape_ev[0] << " " << shape_ev[1] << endl;
	cout << "Correct: 884340 555880" << endl;
	cout << endl;
	cout << "(3) shape pc: " << endl;
	cout << "Yours:   " << shape_pc[0][0].x() << endl;
	cout << "Correct: -0.0024" << endl;
	cout << endl;
	cout << "(1) texture mu: " << endl;
	cout << "Yours:   " << tex_mu[0] << endl;
	cout << "Correct: 182.8750 135.0400 107.1400" << endl;
	cout << endl;
	cout << "(2) texture ev: " << endl;
	cout << "Yours:   " << tex_ev[0] << " " << tex_ev[1] << endl;
	cout << "Correct: 4103.2 2024.1" << endl;
	cout << endl;
	cout << "(3) texture pc: " << endl;
	cout << "Yours:   " << tex_pc[0][0].x() << endl;
	cout << "Correct: -0.0028" << endl;
	cout << endl;
}
