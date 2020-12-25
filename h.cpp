#include <iostream>
#include <Eigen/Dense>
#include <cfloat>
#include <stdarg.h>
#include "ProxGD.h"
#include <string>
using namespace Eigen;
using namespace std;
class Penalty
{
private:
    MatrixXd L1_soft(MatrixXd x, double mu0);
    double mu, alpha, R, constant;
    int R0;
    string mode;
    VectorXd w;
    MatrixXd A, b, L, U;

public:
    Penalty(string mode, int n, ...);
    double h(MatrixXd x);
    MatrixXd prox_h(MatrixXd x);
    int is_positive(MatrixXd x);
    double L_0(MatrixXd x);
    MatrixXd L_0_prox(MatrixXd x);
    double L_1(MatrixXd x);
    MatrixXd L_1_prox(MatrixXd x);
    double L_2(MatrixXd x);
    MatrixXd L_2_prox(MatrixXd x);
    double L_12(MatrixXd x);
    MatrixXd L_12_prox(MatrixXd x);
    double L_21(MatrixXd x);
    MatrixXd L_21_prox(MatrixXd x);
    double L_inf(MatrixXd x);
    MatrixXd L_inf_prox(MatrixXd x);
    double Nuclear(MatrixXd x);
    MatrixXd Nuclear_prox(MatrixXd x);
    double GLasso(MatrixXd x);
    MatrixXd GLasso_prox(MatrixXd x);
    double Log_barrier(MatrixXd x);
    MatrixXd Log_barrier_prox(MatrixXd x);
    double Elastic(MatrixXd x);
    MatrixXd Elastic_prox(MatrixXd x);

    MatrixXd Ind_L_0_prox(MatrixXd x);
    MatrixXd Ind_L_1_prox(MatrixXd x);
    MatrixXd Ind_L_F_prox(MatrixXd x);
    MatrixXd Ind_L_inf_prox(MatrixXd x);
    MatrixXd Ind_box_prox(MatrixXd x);
    MatrixXd Ind_positive_prox(MatrixXd x);
    MatrixXd Ind_negative_prox(MatrixXd x);
    MatrixXd Ind_half_prox(MatrixXd x);
    MatrixXd Ind_affine_prox(MatrixXd x);
    MatrixXd Ind_nuclear_prox(MatrixXd x);
    MatrixXd Ind_psd_prox(MatrixXd x);
    MatrixXd Ind_rank_prox(MatrixXd x);
};

Penalty::Penalty(string mode, int n, ...)
{
    if (mode.substr(0, 3) == "Ind")
    {
        if (n >= 1)
        {
            va_list args;
            va_start(args, n);
            if ((mode == "Ind_rank") || (mode == "Ind_L_0"))
            {
                int R0 = va_arg(args, int);
            }
            else if (mode == "Ind_box")
            {
                MatrixXd L = va_arg(args, MatrixXd);
                MatrixXd U = va_arg(args, MatrixXd);
            }
            else if (mode == "Ind_affine")
            {
                MatrixXd A = va_arg(args, MatrixXd);
                MatrixXd b = va_arg(args, MatrixXd);
            }
            else if (mode == "Ind_affine")
            {
                MatrixXd A = va_arg(args, MatrixXd);
                double constant = va_arg(args, double);
            }
            else
            {
                double R = va_arg(args, double);
            }
        }
    }
    else
    {
        va_list args;
        va_start(args, n);
        double mu = va_arg(args, double);
        if (mode == "Elastic")
            double alpha = va_arg(args, double);
        else if (mode == "GLasso")
            VectorXd w = va_arg(args, VectorXd);
        va_end(args);
    }
}

double Penalty::h(MatrixXd x)
{
    if (mode == "L_0")
        return L_0(x);
    else if (mode == "L_1")
        return L_1(x);
    else if (mode == "L_2")
        return L_2(x);
    else if (mode == "L_12")
        return L_12(x);
    else if (mode == "L_21")
        return L_21(x);
    else if (mode == "L_inf")
        return L_inf(x);
    else if (mode == "Nuclear")
        return Nuclear(x);
    else if (mode == "Log_barrier")
        return Log_barrier(x);
    else if (mode == "Elastic")
        return Elastic(x);
    else if (mode == "GLasso")
        return GLasso(x);
    else
        throw "incorrect objective function.";
}

MatrixXd Penalty::prox_h(MatrixXd x)
{
    if (mode == "L_0")
        return L_0_prox(x);
    else if (mode == "L_1")
        return L_1_prox(x);
    else if (mode == "L_2")
        return L_2_prox(x);
    else if (mode == "L_12")
        return L_12_prox(x);
    else if (mode == "L_21")
        return L_21_prox(x);
    else if (mode == "L_inf")
        return L_inf_prox(x);
    else if (mode == "Nuclear")
        return Nuclear_prox(x);
    else if (mode == "Log_barrier")
        return Log_barrier_prox(x);
    else if (mode == "Elastic")
        return Elastic_prox(x);
    else if (mode == "GLasso")
        return GLasso_prox(x);
    else if (mode == "Ind_L_0")
        return Ind_L_0_prox(x);
    else if (mode == "Ind_L_1")
        return Ind_L_1_prox(x);
    else if (mode == "Ind_L_F")
        return Ind_L_F_prox(x);
    else if (mode == "Ind_L_inf")
        return Ind_L_inf_prox(x);
    else if (mode == "Ind_box")
        return Ind_box_prox(x);
    else if (mode == "Ind_positive")
        return Ind_positive_prox(x);
    else if (mode == "Ind_negative")
        return Ind_negative_prox(x);
    else if (mode == "Ind_half")
        return Ind_half_prox(x);
    else if (mode == "Ind_affine")
        return Ind_affine_prox(x);
    else if (mode == "Ind_nuclear")
        return Ind_nuclear_prox(x);
    else if (mode == "Ind_psd")
        return Ind_psd_prox(x);
    else if (mode == "Ind_rank")
        return Ind_rank_prox(x);
    else
        throw "incorrect Penalty function.";
}

MatrixXd Penalty::L1_soft(MatrixXd x, double mu0)
{
    return (x.array().sign() * (x.array().abs() - mu0).max(0)).matrix();
}

MatrixXd Penalty::Ind_L_0_prox(MatrixXd x)
{
    MatrixXd x1 = x.cwiseAbs();
    Map<VectorXd> v(x1.data(), x1.size());
    std::sort(v.data(), v.data() + v.size());
    double threshold = v(v.size() - R0) - DBL_EPSILON;
    ArrayXXd mask = (x.cwiseAbs().array() > threshold).cast<double>();
    return x.cwiseProduct(mask.matrix());
}

MatrixXd Penalty::Ind_L_1_prox(MatrixXd x)
{
    double l1 = x.cwiseAbs().sum();
    double linf = x.cwiseAbs().maxCoeff();
    if (l1 < R)
    {
        return x;
    }
    double u = linf;
    double l = 0;
    double m = (u + l) / 2.0;
    double vm = (Penalty::L1_soft(x, m)).cwiseAbs().sum();
    while (fabs(vm - R) > FLT_EPSILON)
    {
        if (vm > R)
        {
            l = m;
        }
        else
        {
            u = m;
        }
        m = (u + l) / 2.0;
        vm = (Penalty::L1_soft(x, m)).cwiseAbs().sum();
    }
    return Penalty::L1_soft(x, m);
}

MatrixXd Penalty::Ind_L_F_prox(MatrixXd x)
{
    return x * min(1.0, R / x.norm());
}

MatrixXd Penalty::Ind_L_inf_prox(MatrixXd x)
{
    return x.array().max(-R).min(R).matrix();
}

MatrixXd Penalty::Ind_box_prox(MatrixXd x)
{
    assert(L.rows() == x.rows());
    assert(L.cols() == x.cols());
    assert(U.rows() == x.rows());
    assert(U.cols() == x.cols());
    return (x.array().max(L.array()).min(U.array())).matrix();
}

MatrixXd Penalty::Ind_positive_prox(MatrixXd x)
{
    return x.array().max(0).matrix();
}

MatrixXd Penalty::Ind_negative_prox(MatrixXd x)
{
    return x.array().min(0).matrix();
}

MatrixXd Penalty::Ind_half_prox(MatrixXd x)
{
    assert(A.rows() == x.rows());
    assert(A.cols() == x.cols());
    Map<VectorXd> x1(x.data(), x.size());
    Map<VectorXd> a(A.data(), A.size());
    return x - A / A.norm() * (x1.dot(a) - constant);
}

MatrixXd Penalty::Ind_affine_prox(MatrixXd x)
{
    int n = x.rows(), r = x.cols(), m = A.rows();
    assert(A.cols() == n * r);
    assert(b.rows() == A.rows());
    Map<VectorXd> v(x.data(), x.size());
    VectorXd result = A * v - b;
    MatrixXd ATA = A * A.transpose();
    result = ATA.ldlt().solve(result);
    result = v - A.transpose() * result;
    Map<MatrixXd> Result(result.data(), x.rows(), x.cols());
    return Result;
}

MatrixXd Penalty::Ind_nuclear_prox(MatrixXd x)
{
    MatrixXd result = x;
    JacobiSVD<MatrixXd> svd(x, ComputeThinU | ComputeThinV);
    result = svd.matrixU() * Penalty::Ind_L_1_prox(svd.singularValues()).asDiagonal() * svd.matrixV().transpose();
    return result;
}

MatrixXd Penalty::Ind_psd_prox(MatrixXd x)
{
    assert(x.cols() == x.rows());
    MatrixXd result = x;
    JacobiSVD<MatrixXd> svd((x + x.transpose()) / 2.0, ComputeThinU | ComputeThinV);
    result = svd.matrixU() * (svd.singularValues().array()).max(0).matrix().asDiagonal() * svd.matrixV().transpose();
    return result;
}

MatrixXd Penalty::Ind_rank_prox(MatrixXd x)
{
    JacobiSVD<MatrixXd> svd(x, ComputeThinU | ComputeThinV);
    return svd.matrixU() * svd.singularValues().head(R0).asDiagonal() * svd.matrixV().transpose();
}

double Penalty::L_12(MatrixXd x)
{
    double result = 0.0;
    for (int i = 0; i < x.rows(); i++)
    {
        result += x.row(i).norm();
    }
    return result;
}

double Penalty::L_21(MatrixXd x)
{
    double result = 0.0;
    for (int j = 0; j < x.cols(); j++)
    {
        result += x.col(j).norm();
    }
    return result;
}

double Penalty::Nuclear(MatrixXd x)
{
    JacobiSVD<MatrixXd> svd(x, ComputeThinU | ComputeThinV);
    double result = svd.singularValues().sum();
    return result;
}

double Penalty::L_inf(MatrixXd x)
{
    assert(x.cols() == 1);
    return x.lpNorm<Infinity>();
}

double Penalty::L_0(MatrixXd x)
{
    double result;
    assert(x.cols() == 1);
    for (int i = 0; i < x.rows(); i++)
    {
        result += (fabs(x(i, 0)) > DBL_EPSILON) ? 1 : 0;
    }
    return result;
}

double Penalty::GLasso(MatrixXd x)
{
    assert(w.size() == x.rows());
    double result = 0.0;
    for (int i = 0; i < x.rows(); i++)
    {
        result += x.row(i).norm() * w(i);
    }
    return result;
}

double Penalty::Log_barrier(MatrixXd x)
{
    return -x.array().log().sum();
}

int Penalty::is_positive(MatrixXd x)
{
    for (int i = 0; i < x.rows(); i++)
    {
        for (int j = 0; j < x.cols(); j++)
        {
            if (x(i, j) <= DBL_EPSILON)
            {
                throw "x must be positive.";
            }
        }
    }
    return 0;
}

MatrixXd Penalty::L_12_prox(MatrixXd x)
{
    MatrixXd result = x;
    for (int i = 0; i < x.rows(); i++)
    {
        if (x.row(i).norm() < DBL_EPSILON)
        {
            result.row(i) *= 0;
        }
        double flag = 1 - mu / x.row(i).norm();
        result.row(i) *= max(flag, 0.0);
    }
    return result;
}

MatrixXd Penalty::L_21_prox(MatrixXd x)
{
    MatrixXd result = x;
    for (int i = 0; i < x.cols(); i++)
    {
        if (x.col(i).norm() < DBL_EPSILON)
        {
            result.col(i) *= 0;
        }
        double flag = 1 - mu / x.col(i).norm();
        result.col(i) *= max(flag, 0.0);
    }
    return result;
}

MatrixXd Penalty::Nuclear_prox(MatrixXd x)
{
    MatrixXd result = x;
    JacobiSVD<MatrixXd> svd(x, ComputeThinU | ComputeThinV);
    result = svd.matrixU() * (svd.singularValues().array() - mu).max(0).matrix().asDiagonal() * svd.matrixV().transpose();
    return result;
}

MatrixXd Penalty::L_0_prox(MatrixXd x)
{
    MatrixXd result = x;
    assert(x.cols() == 1);
    double sqrt_mu = sqrt(2 * mu);
    for (int i = 0; i < x.rows(); i++)
    {
        result(i, 0) = (fabs(x(i, 0)) > sqrt_mu) ? x(i, 0) : 0;
    }
    return result;
}

MatrixXd Penalty::L_inf_prox(MatrixXd x)
{
    assert(x.cols() == 1);
    // 确保是向量
    MatrixXd result = x;
    MatrixXd x_abs(x.size() + 1, 1);
    x_abs << x.cwiseAbs(), 0;
    sort(x_abs.data(), x_abs.data() + x_abs.size(), greater<double>());
    // 首先将各元素绝对值按照大小排序，并在结尾补0
    double t = 0;
    // 然后需要确定t，见上机报告(20)式
    double cum = 0;
    int i = 0;
    for (i = 0; (i < x_abs.size() - 1) && (cum < mu); i++)
    {
        cum += (i + 1) * (x_abs(i) - x_abs(i + 1));
    }
    // 确定t所在的区间为[x_abs(i - 1), x_abs(i)]
    if (cum < mu)
    {
        return 0 * result; // 如果mu大于x的l_1范数，直接返回0向量
    }
    t = x_abs(i) + (cum - mu) / i;
    // 计算t的值
    for (int i = 0; i < x.size(); i++)
    {
        if (x(i) > t)
        {
            result(i, 0) = t;
        }
        else if (x(i) < -t)
        {
            result(i, 0) = -t;
        }
    }
    // 修改x的值
    return result;
}

MatrixXd Penalty::GLasso_prox(MatrixXd x)
{
    assert(w.size() == x.rows());
    MatrixXd result = x;
    for (int i = 0; i < x.rows(); i++)
    {
        if (x.row(i).norm() < DBL_EPSILON)
        {
            result.row(i) *= 0;
        }
        double flag = 1 - mu * w(i) / x.row(i).norm();
        result.row(i) *= max(flag, 0.0);
    }
    return result;
}

MatrixXd Penalty::Log_barrier_prox(MatrixXd x)
{
    ArrayXXd xray = x.array();
    return (xray + (xray * xray + 4 * mu).sqrt()).matrix() / 2.0;
}

double Penalty::L_1(MatrixXd x)
{
    assert(x.cols() == 1);
    return Penalty::L_12(x);
}

double Penalty::L_2(MatrixXd x)
{
    assert(x.cols() == 1);
    return Penalty::L_21(x);
}

double Penalty::Elastic(MatrixXd x)
{
    assert((0.0 <= alpha) && (alpha <= 1.0));
    return Penalty::L_12(x) * alpha + (1 - alpha) * x.squaredNorm() / 2.0;
}

MatrixXd Penalty::Elastic_prox(MatrixXd x)
{
    assert((0.0 <= alpha) && (alpha <= 1.0));
    MatrixXd result = x;
    for (int i = 0; i < x.rows(); i++)
    {
        double normx = x.row(i).norm();
        if (normx <= alpha * mu)
        {
            result.row(i) *= 0.0;
        }
        double flag = (1 - alpha * mu / normx) / (1.0 + (1.0 - alpha) * mu);
        result.row(i) *= max(flag, 0.0);
    }
    return result;
}

MatrixXd Penalty::L_1_prox(MatrixXd x)
{
    assert(x.cols() == 1);
    return Penalty::L_12_prox(x);
}

MatrixXd Penalty::L_2_prox(MatrixXd x)
{
    assert(x.cols() == 1);
    return Penalty::L_21_prox(x);
}

int main()
{
    MatrixXd x = MatrixXd::Random(3, 3);
    Penalty h;
    cout << "Mat:" << endl
         << x << endl;
    cout << "Th:" << endl
         << h.Ind_psd_prox(x) << endl;

    return 0;
}