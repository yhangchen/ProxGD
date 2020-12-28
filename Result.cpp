#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "ProxGD.h"
#include <math.h>
using namespace Eigen;
using namespace std;

Result::Result(int iter, MatrixXd x, double min_value) : x(x), iter(iter), min_value(min_value){};
// This is the constructor. It can initialize every memeber variable.

// Result::Result(){};

// Result::~Result(){};

void Result::show()
{
	cout << "min point" << endl;
	cout << x << endl;
	cout << "iterations: " << iter << endl;
	cout << "min value: " << min_value << endl;
	// This function can show each member variable.
}

MatrixXd Result::min_point()
{
	return x; // This function can return the optimal point.
}

double Result::min_loss()
{
	return min_value; // This function can return the optimal value.
}

int Result::iterations()
{
	return iter; // This function can return the time of iterations.
}

int Result::modify_iter(int k)
{
	iter = k;
	return k; // Modify the iter to k.
}
