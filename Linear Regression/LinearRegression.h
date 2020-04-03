

#include<iostream>
#include<Eigen/Dense>

#include "LinearRegressionClass.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

MatrixXd* LinearRegression::get_Data_ptr()
{

	//Returns the pointer to the Data.

	return &Data;

}



VectorXd* LinearRegression::get_Beta_ptr()
{

	//Returns the pointer to the Beta.
	
	return &Beta;

}



VectorXd* LinearRegression::get_YTrue_ptr()
{

	//Returns the pointer to the Ground truth vector.
	
	return &Y_true;

}



// NEXT TASK: CREATE A FUNCTION TO ADD AN INPUT COLUMN AT FIRST INDEX.
//DONE.

MatrixXd LinearRegression::AddOnesColStart(MatrixXd* mat_ptr)
{

	//Returns A NEW MATRIX with the data of mat_ptr but with a column of ONES
	//at the first index of columns.

	MatrixXd ones = MatrixXd::Constant(mat_ptr->rows(), mat_ptr->cols() + 1, 1);

	ones.block(0, 1, mat_ptr->rows(), mat_ptr->cols()) = *mat_ptr;

	return ones;

}


//NEXT TASK : VECTOR-MATRIX MULTIPLICATION OF BETA WITH DATA.

MatrixXd LinearRegression::Predict(MatrixXd* mat_ptr, VectorXd* beta_ptr)
{

	//Multiplies the mat and beta row-wise to obtain prediction for each row of the data.
	//NOTE: It is assumed that the mat_ptr contains data which has a column of ones appended to it.

	MatrixXd result = (*mat_ptr) * (*beta_ptr);
	//result is a vector of shape [n_rows].

	return result;

}


//############################# NEXT GOAL : GRADIENT DESCENT FOR LINEAR REGRESSION ##################### 

//NEXT TASK : Implement Gradient Descent algorithm.

//1. Gradient Calculator: Implement a function which takes the Beta vector and calculates the Gradient of Eucliden Loss Function
// wrt Beta.

VectorXd LinearRegression::Gradient(MatrixXd* X_ptr, VectorXd* Beta_ptr, VectorXd* Y_true_ptr)
{
	// X_ptr : Pointer to the data of shape [n_rows, n_feat + 1], 1st columns is all ones.
	// Beta_ptr : Pointer to the Parameter Vector of shape [n_feat + 1].
	// Y_true_ptr : Pointer to the Ground truth vector of shape [n_rows]

	VectorXd Grad = 2 * (Beta_ptr->transpose() * (X_ptr->transpose() * (*X_ptr))) -  2 * (Y_true_ptr->transpose() * (*X_ptr));

	return Grad;

}


//2. Parameter Updator: Updates the parameters.

VectorXd* LinearRegression::ParameterUpdate(MatrixXd* X_ptr, VectorXd* Beta_ptr, VectorXd* Y_true_ptr, float learning_rate )
{

	// Updates the Parameters IN-PLACE.

	*Beta_ptr = *Beta_ptr - learning_rate*Gradient(X_ptr, Beta_ptr, Y_true_ptr);

	return Beta_ptr;
}




float LinearRegression::LossFunction(VectorXd* y_pred_ptr, VectorXd* y_true_ptr)
{
	//The Euclidean Norm between the predictions and the true values.

	// IMPLEMENT THIS!!

	VectorXd error = (*y_pred_ptr - *y_true_ptr);

	double loss = (error.array().square()).mean();

	return sqrt(loss);
	//It's not exactly Euclidean Norm, but it ain't a big deal.

}






VectorXd* LinearRegression::Trainer(int num_epochs, float learning_rate, int verbose_at = 50)
{


	// Returns the pointer to Private member Beta, which is updated by this function IN-PLACE.


	VectorXd* beta_ptr=nullptr;

	//Step 1 : Add a column of 1s in front of data

	MatrixXd AppendedData = AddOnesColStart(&Data);


	//Step 2 : Start updating the parameters

	for (int i = 1; i<=num_epochs; i++)
	{

		if (i%verbose_at == 0)
		{	
			VectorXd predictions = Predict(&AppendedData, &Beta);

			cout<<"Loss at Epoch # "<<i<<" : "<<LossFunction(&predictions, &Y_true)<<endl<<endl;

		}


		// VectorXd predictions = Predict(&AppendedData, &Beta);
		// cout<<"Loss at Epoch # "<<i<<" : "<<LossFunction(&predictions, &Y_true)<<endl<<endl;

		beta_ptr = ParameterUpdate(&AppendedData, &Beta, &Y_true, learning_rate);

	}

	//Step 3 : Return the pointer to Beta.


	return &Beta;


}


