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

void save_file(string name, string datatype, MatrixXd A)
{
	// Save the matrix A as csv files. name is the file name without ".csv".
	// The file will be in the folder "data".
	stringstream ss;
	ss << "./data/" << name << datatype << ".csv ";
	ss >> name;
	ofstream fout;
	fout.open(name, ios::out | ios::trunc);
	for (int i = 0; i < A.rows(); i++)
	{
		for (int j = 0; j < A.cols(); j++)
		{
			fout << A(i, j) << ' ';
		}
		fout << '\n';
	}
	fout.close();
}

void sparsity_test(string fmode, string hmode, string tmode, int m, int n, int r, double mu, double sp)
{
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
	for (int i = 0; i < ceil(n * sp); i++)
	{
		exact_x.row(a[i]).setRandom();
	}
	Eigen::MatrixXd b = A * exact_x;

	Eigen::MatrixXd x0(n, r);
	x0.setRandom();
	// Initialize A£¬b£¬x0, ecact_x.
	Result res = ProxGD_one_step(fmode, hmode, tmode, &A, &b, x0, mu);
	res.add_exact_x(exact_x);
	res.show();
	// Calculate and show the result.
	string name = fmode + "_" + hmode + "_" + tmode;
	save_file(name, "A", A);
	save_file(name, "b", b);
	save_file(name, "x0", x0);
	save_file(name, "x", res.min_point());
}

void main_parsity_test()
{
	static default_random_engine e(0);
	int m = 1024;
	int n = 512;
	int r = 1;
	double mu = 1e-2;
	double sp = 1e-1; // sparsity.
	string fmode = "Frob";
	string tmode = "BB";
	string hmode = "L_0";
	sparsity_test(fmode, hmode, tmode, m, n, r, mu, sp);
	hmode = "L_1";
	sparsity_test(fmode, hmode, tmode, m, n, r, mu, sp);
	hmode = "L_2";
	sparsity_test(fmode, hmode, tmode, m, n, r, mu, sp);
	hmode = "L_inf";
	sparsity_test(fmode, hmode, tmode, m, n, r, mu, sp);
	hmode = "Elastic";
	sparsity_test(fmode, hmode, tmode, m, n, r, mu, sp);
}

int main()
{
	main_parsity_test();
	return 0;
}
