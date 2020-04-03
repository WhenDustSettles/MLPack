#ifndef LINREG
#define LINREG

#include<iostream>
#include<Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

class LinearRegression
{

private:
	MatrixXd Data; // expected shape : [n_row, n_feat]
	VectorXd Beta; // expected shape : [n_feat + 1]
	VectorXd Y_true; // expected shape : [n_row]

public:
	LinearRegression(MatrixXd* inp_data_ptr = nullptr, VectorXd* init_param_ptr = nullptr, VectorXd* y_true = nullptr)
	{
		//Loads the Data for the class
		if (inp_data_ptr != nullptr){

			Data = *inp_data_ptr;

		}

		if (init_param_ptr != nullptr){

			Beta = *init_param_ptr;

		}

		if (y_true != nullptr){

			Y_true = *y_true;
		}



	}



	MatrixXd* get_Data_ptr();

	
	VectorXd* get_Beta_ptr();


	VectorXd* get_YTrue_ptr();
	

	MatrixXd AddOnesColStart(MatrixXd*);
	

	MatrixXd Predict(MatrixXd* , VectorXd* );
	

	VectorXd Gradient(MatrixXd*, VectorXd*, VectorXd*);
	

	VectorXd* ParameterUpdate(MatrixXd*, VectorXd*, VectorXd* , float );
	

	float LossFunction(VectorXd*, VectorXd* );
	

	VectorXd* Trainer(int , float , int );
	
};

#endif
