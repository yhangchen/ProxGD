#pragma once
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

class Penalty
{
private:
	MatrixXd L1_soft(MatrixXd x, double mu);

public:
	int is_positive(MatrixXd x);
	double L_0(MatrixXd x);
	MatrixXd L_0_prox(MatrixXd x, double mu);
	double L_1(MatrixXd x);
	MatrixXd L_1_prox(MatrixXd x, double mu);
	double L_2(MatrixXd x);
	MatrixXd L_2_prox(MatrixXd x, double mu);
	double L_12(MatrixXd x);
	MatrixXd L_12_prox(MatrixXd x, double mu);
	double L_21(MatrixXd x);
	MatrixXd L_21_prox(MatrixXd x, double mu);
	double L_inf(MatrixXd x);
	MatrixXd L_inf_prox(MatrixXd x, double mu);
	double Nuclear(MatrixXd x);
	MatrixXd Nuclear_prox(MatrixXd x, double mu);
	double GLasso(MatrixXd x, VectorXd w);
	MatrixXd GLasso_prox(MatrixXd x, VectorXd w, double mu);
	double Log_barrier(MatrixXd x);
	MatrixXd Log_barrier_prox(MatrixXd x, double mu);
	double Elastic(MatrixXd x, double alpha);
	MatrixXd Elastic_prox(MatrixXd x, double alpha, double mu);

	MatrixXd Ind_L_0_prox(MatrixXd x, int R);
	MatrixXd Ind_L_1_prox(MatrixXd x, double R);
	MatrixXd Ind_L_F_prox(MatrixXd x, double R);
	MatrixXd Ind_L_inf_prox(MatrixXd x, double R);
	MatrixXd Ind_box_prox(MatrixXd x, MatrixXd L, MatrixXd U);
	MatrixXd Ind_positive_prox(MatrixXd x);
	MatrixXd Ind_negative_prox(MatrixXd x);
	MatrixXd Ind_half_prox(MatrixXd x, MatrixXd A, double alpha);
	MatrixXd Ind_affine_prox(MatrixXd x, MatrixXd A, MatrixXd b);
	MatrixXd Ind_nuclear_prox(MatrixXd x, double R);
	MatrixXd Ind_psd_prox(MatrixXd x);
	MatrixXd Ind_rank_prox(MatrixXd x, int R);
};

class Result
{// This class is used to store and show the result. See Result.cpp.
public:
	Result(int iter, MatrixXd x, double min_value);
	Result();
	~Result();
	void show();// Show the result.
	MatrixXd min_point();
	double min_loss();
	int iterations();
	int modify_iter(int iter);

private:
	MatrixXd x;// Optimal x
	int iter;// Time of iterations
	double min_value;// Optimal value

};

Result ProxGD(string fmode, string hmode, string tmode, MatrixXd A, MatrixXd b, MatrixXd x0, double mu, double epsilon = 1e-6, double gamma = 0.5, int M = 2);
// To calculate the minimum of f(x)+g(x). See ProxGD.cpp.

Result ProxGD_one_step(string fmode, string hmode, string tmode, MatrixXd A, MatrixXd b, MatrixXd x0, double mu, double epsilon = 1e-6, double gamma = 0.5, int M = 2);
// To calculate the minimum of f(x)+g(x). See ProxGD.cpp.

double h(string fmode, MatrixXd x);
// h function in the problem. See h.cpp.

MatrixXd prox_h(string hmode, double mu, MatrixXd x);
// h function in the problem. See h.cpp.

double line_search(Objective& f_obj, string tmode, MatrixXd x, double gamma, int n, ...);
// The function of line searching. It can decide the step size t. See line_search.cpp.