#include <iostream>
#include <Eigen/Dense>
#include "ProxGD.h"
using namespace Eigen;
using namespace std;

class Objective
{
public:
    double Frob(MatrixXd A, MatrixXd b, MatrixXd x);
    MatrixXd Frob_grad(MatrixXd A, MatrixXd b, MatrixXd x);
    double Logistic(MatrixXd A, MatrixXd b, MatrixXd x);
    MatrixXd Logistic_grad(MatrixXd A, MatrixXd b, MatrixXd x);
    int check(MatrixXd A, MatrixXd b, MatrixXd x);
};

double Objective::Frob(MatrixXd A, MatrixXd b, MatrixXd x)
{
    return (A * x - b).squaredNorm() / 2.0;
}

double Objective::Logistic(MatrixXd A, MatrixXd b, MatrixXd x)
{
    assert(x.cols() == 1);
    return (1 + (-(A * x).cwiseProduct(b).array()).exp()).log().sum() / (double)x.rows();
}

MatrixXd Objective::Frob_grad(MatrixXd A, MatrixXd b, MatrixXd x)
{
    return A.transpose() * (A * x - b);
}

MatrixXd Objective::Logistic_grad(MatrixXd A, MatrixXd b, MatrixXd x)
{
    int m = x.rows();
    MatrixXd result = x, Ax = A * x;
    ArrayXd tmp = b.array();
    tmp = (-Ax.array() * tmp).exp();
    tmp *= b.array() / (1.0 + tmp) / (double)m;
    return -A.transpose() * tmp.matrix();
}

int Objective::check(MatrixXd A, MatrixXd b, MatrixXd x)
{
    int m = A.rows(),
        r = A.cols(), r1 = x.rows(), n = x.cols(), m1 = b.rows(), n1 = b.cols();
    if ((m != m1) || (r != r1) || (n != n1))
    {
        throw "Incompatible size.";
    }
    return 0;
}

int main()
{
    MatrixXd A = MatrixXd::Random(3, 3);
    MatrixXd b = MatrixXd::Random(3, 1);
    MatrixXd x = MatrixXd::Random(3, 1);
    Objective f;

    cout
        << "Test:" << endl
        << f.check(A, b, x) << endl;
}