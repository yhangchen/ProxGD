#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <stdarg.h>
#include <stdlib.h>
#include <random>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "ProxGD.h"
#include <assert.h>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <vector>
#include <cfloat>
#include <string>
using namespace Eigen;
using namespace std;

int main()
{
	/*
	static default_random_engine e(1);
	int m = 1024;
	int n = 512;
	int r = 1;

	Eigen::MatrixXd A(m, n);
	A.setRandom();

	int *a = new int[n];
	for (int i = 0; i < n; i++)
	{
		a[i] = i;
	}
	std::shuffle(a, a + n, std::default_random_engine(0));
	
	Eigen::MatrixXd exact_x(n, r);
	exact_x.setZero();
	for (int i = 0; i < ceil(n / 10); i++)
	{
		exact_x.row(a[i]).setRandom();
	}
 
	Eigen::MatrixXd b = A * exact_x;

	Eigen::MatrixXd x0(n, r);
	x0.setRandom();
	// Initialize A£¬b£¬x0, ecact_x.

	Result res = ProxGD_one_step("Frob", "L_2", "BB", A, b, x0, 1e-2);
	res.add_exact_x(exact_x);
	res.show();
	// Calculate and show the result.

	save_file("A", A);
	save_file("b", b);
	save_file("x0", x0);
	save_file("x", res.min_point());
	*/
	
	denoise("fragment1_noise", "fragment1", 44100);
	// denoise("fragment2_noise", "fragment2", 52920);

	return 0;
}