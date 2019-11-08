#include "bfm.h"

bool init_bfm() {
	for (int i = 0; i < N_PC; i++) {
		shape_pc[i].resize(N_VERTICE);
		tex_pc[i].resize(N_VERTICE);
	}
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
	return true;
}


void generate_random_face(double scale) {
	cout << "generate random face" << endl;
	cout << "	generate random sequence - ";
	alpha = randn(N_PC, scale);
	beta = randn(N_PC, scale);
	cout << "success" << endl;
	generate_face();
}

void generate_face() {
	cout << "	pca - ";
	shape = coef2object(alpha, shape_mu, shape_pc, shape_ev);
	tex = coef2object(beta, tex_mu, tex_pc, tex_ev);
	cout << "success" << endl;
}

void save_ply(string filename) {
	cout << "	write into .ply file - ";
	ply_write("rnd_face.ply");
	cout << "success" << endl;
}


std::vector<point3f> coef2object(std::vector<double> &coef, std::vector<point3f> &mu, std::vector<vector<point3f>> &pc, std::vector<double> &ev) {
	std::vector<double> temp = dot(coef, ev);
	return mu + pc * temp;
}

void ply_write(string fn) {
	ofstream out;
	out.open(fn, std::ios::binary);
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
		float x = float(shape[i].x());
		float y = float(shape[i].y());
		float z = float(shape[i].z());
		unsigned char r = tex[i].x();
		unsigned char g = tex[i].y();
		unsigned char b = tex[i].z();
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

