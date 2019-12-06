#include "random.h"

dlib::matrix<double> randn(int n, double scale) {
	std::random_device rd;
	std::mt19937 gen(rd());
	dlib::matrix<double> res(n, 1);
	std::normal_distribution<double> dis(0, scale);
	for (int i = 0; i < n; i++)
		res(i) = dis(gen);
	return res;
}