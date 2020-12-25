#include <iostream>
#include <Eigen/Dense>
#include "ProxGD.h"
#include <string>
using namespace Eigen;
using namespace std;

class Objective
{
public:
    Objective(string mode, MatrixXd A, MatrixXd b);
    double f(MatrixXd x);
    MatrixXd grad_f(MatrixXd x);
    double Frob(MatrixXd x);
    MatrixXd Frob_grad(MatrixXd x);
    double Logistic(MatrixXd x);
    MatrixXd Logistic_grad(MatrixXd x);
    int check(MatrixXd x);

private:
    string mode;
    MatrixXd A;
    MatrixXd b;
};

Objective::Objective(string mode, MatrixXd A, MatrixXd b) : mode(mode), A(A), b(b){};

double Objective::f(MatrixXd x)
{
    if (mode == "Frob")
    {
        return Frob(x);
    }
    else if (mode == "Logistic")
    {
        return Logistic(x);
    }
    else
    {
        throw "incorrect objective function.";
    }
}
MatrixXd Objective::grad_f(MatrixXd x)
{
    if (mode == "Frob")
    {
        return Frob_grad(x);
    }
    else if (mode == "Logistic")
    {
        return Logistic_grad(x);
    }
    else
    {
        throw "incorrect objective function.";
    }
};

double Objective::Frob(MatrixXd x)
{
    return (A * x - b).squaredNorm() / 2.0;
}

double Objective::Logistic(MatrixXd x)
{
    assert(x.cols() == 1);
    return (1 + (-(A * x).cwiseProduct(b).array()).exp()).log().sum() / (double)x.rows();
}

MatrixXd Objective::Frob_grad(MatrixXd x)
{
    return A.transpose() * (A * x - b);
}

MatrixXd Objective::Logistic_grad(MatrixXd x)
{
    int m = x.rows();
    MatrixXd result = x, Ax = A * x;
    ArrayXd tmp = b.array();
    tmp = (-Ax.array() * tmp).exp();
    tmp *= b.array() / (1.0 + tmp) / (double)m;
    return -A.transpose() * tmp.matrix();
}

int Objective::check(MatrixXd x)
{
    int m = A.rows(),
        r = A.cols(), r1 = x.rows(), n = x.cols(), m1 = b.rows(), n1 = b.cols();
    if ((m != m1) || (r != r1) || (n != n1))
    {
        throw "Incompatible size.";
    }
    return 0;
}

void main()
{
    // MatrixXd A = MatrixXd::Random(3, 3);
    // MatrixXd b = MatrixXd::Random(3, 1);
    // MatrixXd x = MatrixXd::Random(3, 1);
    // Objective f(1, A, b);

    // cout
    //     << "Test:" << endl
    //     << f.check(x) << endl;
}