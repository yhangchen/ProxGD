#include <iostream>
#include <Eigen/Dense>
#include "ProxGD.h"
#include <math.h>
using namespace Eigen;
using namespace std;
class Result
{
public:
    Result();
    ~Result();
    void show(); // show用来展示计算结果，目前打算展示iter和min_value
    MatrixXf min_point();

private:
    MatrixXf x;       // 最优的x
    int iter;         // 迭代次数
    double min_value; // 函数的最小值
};

Result::Result()
{
}

Result::~Result()
{
}

void Result::show()
{
}

MatrixXf Result::min_point()
{
    return MatrixXf();
}
