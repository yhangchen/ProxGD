#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <stdarg.h>
#include <stdlib.h>
#include <random>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "ProxGD.h"
#include <assert.h>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <vector>
#include <cfloat>
#include <string>
using namespace Eigen;
using namespace std;

double denoise(string name1, string name2, int row, int col)
{
    string path1;
    string path2;
    string path3;
    MatrixXd A(row, row);
    A.setIdentity();
    cout << "Denoise" << endl;
    for (int i = 0; i < 3; i++)
    {

        path1 = "./image/" + name1 + to_string(i + 1) + ".csv";
        MatrixXd x0 = read_mat(row, col, path1);
        path2 = "./image/" + name2 + to_string(i + 1) + ".csv";
        MatrixXd x_exact = read_mat(row, col, path2);
        cout << x_exact << endl;
        Result res = ProxGD_one_step("Frob", "TV_2D", "BB", &A, &x_exact, x0, 1e-2);
        res.show();
        path3 = name2 + "_denoise_" + to_string(i + 1) + ".csv";
        // save_file(path3, res.min_point());
    }

    return 0.0;
}

MatrixXd read_mat(int row, int col, string name)
{
    MatrixXd A(row, col);

    std::ifstream indata;
    indata.open(name);

    std::string line;
    std::getline(indata, line);
    for (int i = 0; i < row; i++)
    {
        std::getline(indata, line);
        for (int j = 0; j < col; j++)
        {
            std::stringstream lineStream(line);
            std::string cell;

            while (std::getline(lineStream, cell, ','))
            {
                A(i, j) = std::stod(cell);
            }
        }
    }

    indata.close();
    return A;
}

void save_file(string name, string datatype, MatrixXd A)
{
    // Save the matrix A as csv files. name is the file name without ".csv".
    // The file will be in the folder "data".
    stringstream ss;
    ss << "./data/" << name << datatype << ".csv ";
    ss >> name;
    ofstream fout;
    fout.open(name, ios::out | ios::trunc);
    for (int i = 0; i < A.rows(); i++)
    {
        for (int j = 0; j < A.cols(); j++)
        {
            fout << A(i, j) << ' ';
        }
        fout << '\n';
    }
    fout.close();
}

void sparsity_test(string fmode, string hmode, string tmode, int m, int n, int r, double mu, double sp)
{
    Eigen::MatrixXd A(m, n);
    A.setRandom();

    int *a = new int[n];
    for (int i = 0; i < n; i++)
    {
        a[i] = i;
    }
    std::shuffle(a, a + n, std::default_random_engine(0));

    Eigen::MatrixXd exact_x(n, r);
    exact_x.setZero();
    for (int i = 0; i < ceil(n * sp); i++)
    {
        exact_x.row(a[i]).setRandom();
    }
    Eigen::MatrixXd b = A * exact_x;

    Eigen::MatrixXd x0(n, r);
    x0.setRandom();
    // Initialize A£¬b£¬x0, ecact_x.
    Result res = ProxGD_one_step(fmode, hmode, tmode, &A, &b, x0, mu);
    res.add_exact_x(exact_x);
    res.show();
    // Calculate and show the result.
    string name = fmode + "_" + hmode + "_" + tmode;
    save_file(name, "_A", A);
    save_file(name, "_b", b);
    save_file(name, "_x0", x0);
    save_file(name, "_x", res.min_point());
}

void main_parsity_test()
{
    static default_random_engine e(0);
    int m = 1024;
    int n = 512;
    int r = 1;
    double mu = 1e-2;
    double sp = 1e-1; // sparsity.
    string fmode = "Frob";
    string tmode = "BB";
    string hmode = "L_0";
    cout << hmode << endl;
    sparsity_test(fmode, hmode, tmode, m, n, r, mu, sp);
    hmode = "L_1";
    cout << hmode << endl;
    sparsity_test(fmode, hmode, tmode, m, n, r, mu, sp);
    hmode = "L_2";
    cout << hmode << endl;
    sparsity_test(fmode, hmode, tmode, m, n, r, mu, sp);
    hmode = "L_inf";
    cout << hmode << endl;
    sparsity_test(fmode, hmode, tmode, m, n, r, mu, sp);
    hmode = "Elastic";
    cout << hmode << endl;
    sparsity_test(fmode, hmode, tmode, m, n, r, mu, sp);
    hmode = "TV_1D";
    cout << hmode << endl;
    sparsity_test(fmode, hmode, tmode, m, n, r, mu, sp);
    r = 4;
    hmode = "L_12";
    cout << hmode << endl;
    sparsity_test(fmode, hmode, tmode, m, n, r, mu, sp);
    hmode = "L_21";
    cout << hmode << endl;
    sparsity_test(fmode, hmode, tmode, m, n, r, mu, sp);
}

int main()
{
    main_parsity_test();
    // denoise("img_noise", "img", 512, 512);
    return 0;
}
