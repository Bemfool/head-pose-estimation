#include "random.h"

double *randn(int n, double scale) {
	std::random_device rd;
	std::mt19937 gen(rd());
	double *res = new double[n];
	std::normal_distribution<double> dis(0, scale);
	for (int i = 0; i < n; i++)
		res[i] = dis(gen);
	return res;
}