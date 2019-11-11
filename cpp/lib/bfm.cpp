#include "bfm.h"

bool init_bfm() {
	std::cout << "init basel face model ..." << std::endl;
	if (load() == FAIL) {
		cout << "failed to load all data, and errors are listed above." << endl;
		cout << "continue or not? [Y/n] ";
		char option;
		while (true) {
			cin >> option;
			if (option == 'Y' || option == 'y')
				break;
			else if (option == 'N' || option == 'n') {
				return false;
			}
			else
				cout << "please input 'Y'/'y' or 'N'/'n'." << endl;
		}
	}
	else {
		cout << "load data - success" << endl;
	}
	generate_random_face(0.0);
	load_landmark_idx();
	return true;
}


void generate_random_face(double scale) {
	std::cout << "generate random face" << std::endl;
	std::cout << "	generate random sequence - ";
	shape_coef = randn(N_PC, scale);
	tex_coef   = randn(N_PC, scale);
	std::cout << "success" << std::endl;
	generate_face();
}

void generate_face() {
	std::cout << "	pca - ";
	current_shape = coef2object(shape_coef, shape_mu, shape_pc, shape_ev);
	current_tex = coef2object(tex_coef, tex_mu, tex_pc, tex_ev);
	std::cout << "success" << std::endl;
}

void load_landmark_idx() {
	std::cout << "	load landmark index - ";
	ifstream in;
	in.open("../data/68-bfm-landmark.anl");
	if(!in) {
		std::cout << "fail" << std::endl;
		return;
	}
	for(int i=0; i<LANDMARK_NUM; i++)
		in >> landmark_idx[i];
	std::cout << "success" << std::endl;
}

dlib::matrix<double> coef2object(dlib::matrix<double> &coef, 
								 dlib::matrix<double> &mu, 
								 dlib::matrix<double> &pc, 
								 dlib::matrix<double> &ev) {
	return mu + pc * pointwise_multiply(coef, ev);
}

void ply_write(string fn) {
	ofstream out;
	/* Note: In Linux Cpp, we should use std::ios::out as flag, which is not necessary in Windows */
	out.open(fn, std::ios::out | std::ios::binary);
	if (!out) {
		std::cout << "Creation of " << fn << " failed." << std::endl;
		return;
	}
	out << "ply\n";
	out << "format binary_little_endian 1.0\n";
	out << "comment Made from the 3D Morphable Face Model of the Univeristy of Basel, Switzerland.\n";
	out << "element vertex " << N_VERTICE << "\n";
	out << "property float x\n";
	out << "property float y\n";
	out << "property float z\n";
	out << "property uchar red\n";
	out << "property uchar green\n";
	out << "property uchar blue\n";
	out << "element face " << N_FACE << "\n";
	out << "property list uchar int vertex_indices\n";
	out << "end_header\n";

	for (int i = 0; i < N_VERTICE; i++) {
		float x = float(current_shape(i * 3    ));
		float y = float(current_shape(i * 3 + 1));
		float z = float(current_shape(i * 3 + 2));
		unsigned char r = current_tex(i * 3    );
		unsigned char g = current_tex(i * 3 + 1);
		unsigned char b = current_tex(i * 3 + 2);
		out.write((char *)&x, sizeof(x));
		out.write((char *)&y, sizeof(y));
		out.write((char *)&z, sizeof(z));
		out.write((char *)&r, sizeof(r));
		out.write((char *)&g, sizeof(g));
		out.write((char *)&b, sizeof(b));
	}

	unsigned char N_VER_PER_FACE = 3;
	for (int i = 0; i < N_FACE; i++) {
		out.write((char *)&N_VER_PER_FACE, sizeof(N_VER_PER_FACE));
		int x = tl[i].x();
		int y = tl[i].y();
		int z = tl[i].z();
		out.write((char *)&y, sizeof(y));
		out.write((char *)&x, sizeof(x));
		out.write((char *)&z, sizeof(z));
	}
	out.close();
}

