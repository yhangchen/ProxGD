#include <iostream>
#include <stdarg.h>
#include <Eigen/Dense>
#include "ProxGD.h"
#include <assert.h>
#include <algorithm>
using namespace Eigen;
using namespace std;

double line_search(int fmode, int tmode, MatrixXd A, MatrixXd b, MatrixXd x, double gamma, int n, ...)
{
    // 线搜索函数，用于决定算法中的步长t，tmode表示线搜索的方式，A是参数，返回步长t
    // 注：这里线搜索函数没有写完，输入可能会改动，如果用Armijo准则的话肯定要加其他参数

    double t;

    if (tmode == 1)
    {
        // tmode=1的时候取Armijo准则

        va_list args;
        va_start(args, n);
        double fh = va_arg(args, double);
        va_end(args);

        double l_t = 0;
        double r_t = 1;
        t = 1;
        bool flag = 1; // flag = 1表示继续扩大搜索区间
        bool cond;     // cond表示Armijo条件是否已经满足

        while (r_t - l_t > 1e-8)
        {
            // 如果区间长度很小则停止搜索

            cond = (f(fmode, A, b, x - t * grad_f(fmode, A, b, x)) < f(fmode, A, b, x) - gamma * t * grad_f(fmode, A, b, x).squaredNorm()); // 检验Armijo条件是否已经满足

            if (cond)
            {
                // 如果满足，检查是否需要继续扩大搜索区间
                if (flag)
                {
                    l_t = r_t;
                    r_t *= 2;
                    t = r_t;
                    // 如果需要则继续扩大搜索区间
                }
                else
                {
                    break;
                    // 否则直接break
                }
            }
            else
            {
                // 如果不满足，开始缩小区间

                flag = 0; // 首先将flag设为0，防止继续扩大区间

                r_t = t;
                t = (r_t + l_t) / 2; // 然后缩小区间，调整步长
            }
        }
    }
    else if (tmode == 2)
    {
        // tmode=2的时候取BB步长

        va_list args;
        va_start(args, n);
        double *fhs = va_arg(args, double *);
        int M = va_arg(args, int);
        MatrixXd delta_x = va_arg(args, MatrixXd);
        va_end(args);

        MatrixXd delta_g = grad_f(fmode, A, b, x) - grad_f(fmode, A, b, x - delta_x);
        t = delta_x.squaredNorm() / (delta_x.array().cwiseProduct(delta_g.array())).sum();

        while (f(fmode, A, b, x - t * grad_f(fmode, A, b, x)) > *(max_element(fhs, fhs + M)) - gamma * t * grad_f(fmode, A, b, x).squaredNorm())
        {
            // 如果不满足条件则缩小步长

            t /= 2;
        }
    }
    return t;
}