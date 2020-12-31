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

double denoise(string name1, string name2, int row)
{
	string path1;
	string path2;
	string path3;
	stringstream ss;
	SparseMatrix<double> A(row, row);
	A.setIdentity();
	for (int i = 0; i < 3; i++)
	{
		path1 = "";
		path2 = "";
		path3 = "";
		ss.str("");
		ss.clear();
		ss << "data\\" << name1 << i + 1 << ".csv";
		ss >> path1;
		MatrixXd x0 = read_mat(row, 1, path1);
		ss.str("");
		ss.clear();
		ss << "data\\" << name2 << i + 1 << ".csv";
		ss >> path2;
		MatrixXd x_exact = read_mat(row, 1, path2);
		Result res = ProxGD_one_step("Frob", "TV_1D", "BB", A, x_exact, x0, 1e-2);
		res.show();
		ss.str("");
		ss.clear();
		ss << name1 << "_denoised" << i + 1;
		ss >> path3;
		save_file(path3, res.min_point());
	}
	
	return 0.0;
}

MatrixXd read_mat(int row, int col, string name)
{
	MatrixXd A(row, col);
	A.setZero();

	std::ifstream indata;
	indata.open(name);

	std::string line;
	std::stringstream lineStream;
	std::string  cell;
	std::getline(indata, line);
	for (int i = 0; i < row; i++)
	{
		if (std::getline(indata, line))
		{
			lineStream.str("");
			lineStream.clear();
			lineStream << line;
			for (int j = 0; j < col; j++)
			{
				if (std::getline(lineStream, cell, ','))
				{
					A(i, j) = std::stod(cell);
					cell = "";
				}
				else
				{
					cout << "This line is:" << endl;
					cout << line << endl;
					cout << "The matrix is not wide enough." << endl;
					cout << "Now j = " << j << "." << endl;
					cout << "Now i = " << i << "." << endl;
					cout << "The name is: " << name << "." << endl;
				}
			}
		}
		else
		{
			cout << "The matrix is not high enough." << endl;
			cout << "Now i = " << i << "." << endl;
			cout << "The name is: " << name << "." << endl;
		}	
	}

	indata.close();
	return A;
}

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