#include <iostream>
#include <Eigen/Dense>
#include "ProxGD.h"
#include <math.h>
#include <assert.h>
using namespace Eigen;
using namespace std;

double ProxGD(int fmode, int hmode, int tmode, MatrixXd A, MatrixXd b, MatrixXd x0, double mu, double epsilon, double gamma, int M)
{
	// 用于计算f(x)+g(x)的最小值，输入值fmode、hmode、tmode分别决定f、h以及先搜索的模式（分别对应题目中的第三点、
	// 第四点、第二点）。A，b是f的参数，x0是迭代初始值，mu是h前面的系数，epsilon是迭代终止条件的参数。返回值目前
	// 是函数最小值，后面应该会改成Result类，同时返回迭代次数，最小值以及最优的x。

	MatrixXd new_x = x0;
	MatrixXd prev_x = x0;
	MatrixXd delta_x = x0;
	MatrixXd x_star = x0;
	delta_x.setOnes();
	// 初始化变量，在一次循环中会涉及new_x以及prev_x，分别表示新的x和旧的x，delta_x为二者的差，x_star为算法第一步

	int iter = 0;
	double t;
	double *fhs = new double[M];
	fhs[0] = f(fmode, A, b, new_x) + mu * h(hmode, new_x);
	for (int i = 1; i < M; i++)
	{
		fhs[i] = fhs[0];
	}
	fhs[M - 1]++;
	// 初始化变量，t是每次迭代的步长，iter是迭代次数。fhs是前M步的函数值，每次迭代更新一个元素，iter % M表示当前函
	// 数值，往前推是前面的函数值，第0位往前推是第M - 1位，初始化第0位和第M - 1位保证能进入循环。

	while ((fabs(fhs[iter % M] - fhs[(iter + M - 1) % M]) > epsilon) || (delta_x.squaredNorm() > epsilon))
	{
		// 迭代终止条件为函数值变化或者自变量的变化小于等于epsilon（这里可以看出之前初始化fhs[M - 1]和delta_x时的目的）

		if (tmode == 1)
		{
			t = line_search(fmode, tmode, A, b, new_x, gamma, 1, (double)fhs[iter % M]);
		}
		else if (tmode == 2)
			12;
		{
			t = line_search(fmode, tmode, A, b, new_x, gamma, 3, fhs, M, delta_x);
		}
		// 首先利用线搜索确定步长

		x_star = new_x - t * grad_f(fmode, A, b, new_x);
		// 然后计算中间变量x_star

		prev_x = new_x;
		new_x = prox_h(hmode, t * mu, x_star); // 利用prox_h计算新的x
		delta_x = new_x - prev_x;
		// 更新prev_x后再更新new_x，然后计算新的delta_x

		iter = iter + 1;
		// 更新迭代次数

		fhs[iter % M] = f(fmode, A, b, new_x) + mu * h(hmode, new_x);
		;
		// 更新fhs；
	}

	return fhs[iter % M];
}
