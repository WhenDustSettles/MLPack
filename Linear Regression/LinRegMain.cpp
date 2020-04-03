#include<iostream>
#include<Eigen/Dense>

#include "LinearRegression.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;



int main(){

	int n_rows = 44;
	int n_feat = 5;


	//########### TESTING ON DUMMY DATA ################

	MatrixXd m = MatrixXd::Random(n_rows, n_feat);
	VectorXd y_true = VectorXd::Random(n_rows);
	VectorXd b_init = VectorXd::Constant(n_feat + 1, 0);

	LinearRegression linreg(&m, &b_init, &y_true);

	linreg.Trainer(10000,0.0005);

	cout<<*linreg.get_Beta_ptr()<<endl;

}