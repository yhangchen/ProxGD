#pragma once
using namespace Eigen;
using namespace std;

class Result
{ // Result类用来返回最后的计算结果，其内容包括最优的x，函数的最小值以及迭代次数。（这个类还没有开始写）
public:
    Result();
    ~Result();
    void show(); // show用来展示计算结果，目前打算展示iter和min_value
    MatrixXd min_point();

private:
    MatrixXd x;       // 最优的x
    int iter;         // 迭代次数
    double min_value; // 函数的最小值
};

double ProxGD(int fmode, int hmode, int tmode, MatrixXd A, MatrixXd b, MatrixXd x0, double mu, double epsilon, double gamma, int M);
// 用于计算f(x)+g(x)的最小值，详细介绍见ProxGD.cpp

double f(int fmode, MatrixXd A, MatrixXd b, MatrixXd x);
// 问题中的f(x)，详细介绍见f.cpp

MatrixXd grad_f(int fmode, MatrixXd A, MatrixXd b, MatrixXd x);
// 问题中的f(x)的梯度，详细介绍见f.cpp

double h(int fmode, MatrixXd x);
// 问题中的h(x)，详细介绍见h.cpp

MatrixXd prox_h(int hmode, double mu, MatrixXd x);
// 问题中的h(x)的梯度，详细介绍见h.cpp

double line_search(int fmode, int tmode, MatrixXd A, MatrixXd b, MatrixXd x, double gamma, int n, ...);
// 线搜索函数，用于决定算法中的步长t，详细介绍见line_search.cpp