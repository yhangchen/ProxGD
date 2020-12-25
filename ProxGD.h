#pragma once
using namespace Eigen;
using namespace std;

class Result
{ // Result�������������ļ������������ݰ������ŵ�x����������Сֵ�Լ�����������������໹û�п�ʼд��
public:
	Result(int iter, MatrixXd x, double min_value);
	Result();
	~Result();
	void show(); // show����չʾ��������Ŀǰ����չʾiter��min_value
	MatrixXd min_point();
	double min_loss();
	int iterations();
	int modify_iter(int iter);

private:
	MatrixXd x;		  // ���ŵ�x
	int iter;		  // ��������
	double min_value; // ��������Сֵ
};

Result ProxGD(int fmode, int hmode, int tmode, MatrixXd A, MatrixXd b, MatrixXd x0, double mu, double epsilon, double gamma, int M);
// ���ڼ���f(x)+g(x)����Сֵ����ϸ���ܼ�ProxGD.cpp

Result ProxGD_one_step(int fmode, int hmode, int tmode, MatrixXd A, MatrixXd b, MatrixXd x0, double mu, double epsilon, double gamma, int M);
// ���ڼ���f(x)+g(x)����Сֵ����ϸ���ܼ�ProxGD.cpp

/*
double f(int fmode, MatrixXd A, MatrixXd b, MatrixXd x);
// �����е�f(x)����ϸ���ܼ�f.cpp

MatrixXd grad_f(int fmode, MatrixXd A, MatrixXd b, MatrixXd x);
// �����е�f(x)���ݶȣ���ϸ���ܼ�f.cpp
*/

double h(int fmode, MatrixXd x);
// �����е�h(x)����ϸ���ܼ�h.cpp

MatrixXd prox_h(int hmode, double mu, MatrixXd x);
// �����е�h(x)���ݶȣ���ϸ���ܼ�h.cpp

double line_search(Objective &f_obj, int tmode, MatrixXd x, double gamma, int n, ...);
// ���������������ھ����㷨�еĲ���t����ϸ���ܼ�line_search.cpp