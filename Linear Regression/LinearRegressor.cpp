// Class oriented implementation of Linear Regression
// Input to the whole program should be the data of the following format:-
// 


//Linear Regression fits a linear model with coefficients w = (w1, …, wp) to minimize the residual sum of 
//squares between the observed targets in the dataset, and the targets predicted by the linear approximation.
//
// y_i = b0 + b1*x1i + b2*x2i + .... + bp*xpi
//or,
//Y = X.T \dot B + B0
//or, by putting B0 in B and adding a column of 1's as first column in X, we get a simple form:-
//Y = X.T * B
//
//data : Matrix of shape [n_rows, 1 + n_feat]
//							N           P
//We'll use Gradient Descent to find optimal Parameter vector B
//
//


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

	MatrixXd* get_Data_ptr()
	{

		//Returns the pointer to the Data.
	
		return &Data;

	}


	
	VectorXd* get_Beta_ptr()
	{

		//Returns the pointer to the Beta.
		
		return &Beta;

	}



	VectorXd* get_YTrue_ptr()
	{

		//Returns the pointer to the Ground truth vector.
		
		return &Y_true;

	}



	// NEXT TASK: CREATE A FUNCTION TO ADD AN INPUT COLUMN AT FIRST INDEX.
	//DONE.

	MatrixXd AddOnesColStart(MatrixXd* mat_ptr)
	{

		//Returns A NEW MATRIX with the data of mat_ptr but with a column of ONES
		//at the first index of columns.

		MatrixXd ones = MatrixXd::Constant(mat_ptr->rows(), mat_ptr->cols() + 1, 1);

		ones.block(0, 1, mat_ptr->rows(), mat_ptr->cols()) = *mat_ptr;

		return ones;

	}


	//NEXT TASK : VECTOR-MATRIX MULTIPLICATION OF BETA WITH DATA.

	MatrixXd Predict(MatrixXd* mat_ptr, VectorXd* beta_ptr)
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

	VectorXd Gradient(MatrixXd* X_ptr, VectorXd* Beta_ptr, VectorXd* Y_true_ptr)
	{
		// X_ptr : Pointer to the data of shape [n_rows, n_feat + 1], 1st columns is all ones.
		// Beta_ptr : Pointer to the Parameter Vector of shape [n_feat + 1].
		// Y_true_ptr : Pointer to the Ground truth vector of shape [n_rows]

		VectorXd Grad = 2 * (Beta_ptr->transpose() * (X_ptr->transpose() * (*X_ptr))) -  2 * (Y_true_ptr->transpose() * (*X_ptr));

		return Grad;

	}


	//2. Parameter Updator: Updates the parameters.

	VectorXd* ParameterUpdate(MatrixXd* X_ptr, VectorXd* Beta_ptr, VectorXd* Y_true_ptr, float learning_rate )
	{

		// Updates the Parameters IN-PLACE.

		*Beta_ptr = *Beta_ptr - learning_rate*Gradient(X_ptr, Beta_ptr, Y_true_ptr);

		return Beta_ptr;
	}




	float LossFunction(VectorXd* y_pred_ptr, VectorXd* y_true_ptr)
	{
		//The Euclidean Norm between the predictions and the true values.

		// IMPLEMENT THIS!!

		VectorXd error = (*y_pred_ptr - *y_true_ptr);

		double loss = (error.array().square()).mean();

		return sqrt(loss);
		//It's not exactly Euclidean Norm, but it ain't a big deal.

	}






	VectorXd* Trainer(int num_epochs, float learning_rate, int verbose_at = 50)
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


};


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














	// cout<<"M:"<<endl<<m<<endl<<endl;

	// cout<<"b:"<<endl<<b<<endl<<endl;

	//cout<<*linreg.get_Data_ptr()<<endl<<endl;

	//cout<<linreg.MatMultiplyXB(&m, &v)<<endl<<endl;

	// MatrixXd m_1 = linreg.AddOnesColStart(&m);

	// cout<<"M_1:"<<endl<<m_1<<endl<<endl;


	// cout<<"Gradient: "<<endl<<linreg.Gradient(&m_1, &b, &y_true)<<endl<<endl;

	//cout<<linreg.Gradient(&m, &b, &y_true)<<endl<<endl;

	//cout<<b.transpose() - y_true.transpose() * m<<endl;

	// VectorXd test_a = VectorXd::Constant(n_feat + 1,99);

	// VectorXd test_b = VectorXd::Constant(n_feat + 1,25);

	// cout<<"loss: "<<linreg.LossFunction(&test_a, &test_b)<<endl<<endl;

	// cout<<"Predictions:- "<<endl<<linreg.Predict(&m_1, &b)<<endl<<endl;



	//cout<<"New Parameters (b): "<<endl<<*linreg.ParameterUpdate(&m_1, &b, &y_true, 0.01);

}
