#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <stdarg.h>
#include <stdlib.h>
#include <random>
#include <Eigen/Dense>
#include "ProxGD.h"
#include <assert.h>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cstring>
using namespace Eigen;
using namespace std;

void save_file(string name, MatrixXd A)
{
	// Save the matrix A as csv files. name is the file name without ".csv".
	// The file will be in the folder "data".
	stringstream ss;
	ss << "data\\" << name << ".csv";
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

int main()
{
	static default_random_engine e(1);
	int m = 100;
	int n = 100;
	int r = 2;

	Eigen::MatrixXd A(m, n);
	A.setRandom();

	Eigen::MatrixXd b(m, r);
	b.setRandom();

	Eigen::MatrixXd x0(n, r);
	x0.setRandom();
	// Initialize A, b, x0.

	string fmode = "Frob";
	string hmode = "L_12";
	string tmode = "BB";

	Result res = ProxGD_one_step(fmode, hmode, tmode, A, b, x0, 1e-2);
	res.show();
	// Calculate and show the result.

	save_file("A", A);
	save_file("b", b);
	save_file("x0", x0);
	save_file("x", res.min_point());

	return 0;
}