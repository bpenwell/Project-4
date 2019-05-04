#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <string>
#include <queue>
#include <sys/stat.h>
#include <dirent.h>
#include <sstream>
#include <limits>	// numeric_limits
#include <algorithm> // fill
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Eigenvalues" // used to decompose matricies
#include "libsvm-3.23/svm.h"

using Eigen::VectorXd;
using Eigen::VectorXi;
using Eigen::VectorXcf;
using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::MatrixXcd;
using Eigen::MatrixXcf;
using Eigen::EigenSolver;
using namespace std;

//Created by Ben Penwell and Adam Landis
//Pattern Recognition, Project 4
//May 7, 2019

const int K_VAL = 30;
const int NUM_FOLDS = 3;

class DataStorage
{
public:
	DataStorage(string dataSet, string dbRoot = "genderdata");

	void Init();
	void ReadFoldValues(string setPfx, int foldNum);
	void ReadFoldLabels(string setPfx, int foldNum);
	void ReduceSizeTo30();
	void ComputeAvgEigenFeatures();
	void DblVectorToE3Matrix(vector<double> input, bool isMaleVector, bool convertVariance, int fold);
	void BayesianClassifier(int fold);

	const string m_DATA_PATH;
	const string m_EXT;

	const string m_TG_PFX = "T";	// Target filename prefix
	const string m_TR_PFX = "tr";	// Training filename prefix
	const string m_VAL_PFX = "val";	// Validation filename prefix
	const string m_TS_PFX = "ts";	// Test filename prefix

	const int TR_FOLD_SIZE = 134;
	const int VAL_FOLD_SIZE = 133;
	const int TS_FOLD_SIZE = 133;

	vector<vector<double> > m_TrainVector[NUM_FOLDS];
	vector<int> m_TrainTargetVector[NUM_FOLDS];

	vector<vector<double> > m_ValidateVector[NUM_FOLDS];
	vector<int> m_ValidateTargetVector[NUM_FOLDS];

	vector<vector<double> > m_TestVector[NUM_FOLDS];
	vector<int> m_TestTargetVector[NUM_FOLDS];

	vector<double> m_MeanMaleEigenFeatures[NUM_FOLDS];
	vector<double> m_MeanFemaleEigenFeatures[NUM_FOLDS];

	MatrixXd m_Male_MU[NUM_FOLDS];
	MatrixXd m_Female_MU[NUM_FOLDS];

	vector<double> m_VarMaleEigenFeatures[NUM_FOLDS];
	vector<double> m_VarFemaleEigenFeatures[NUM_FOLDS];

	MatrixXd m_Male_VAR[NUM_FOLDS];
	MatrixXd m_Female_VAR[NUM_FOLDS];
};

void PrintVector2D(vector<vector<double> > vector);
void PrintVector(vector<int> vector);
void ExtractAVG(
	vector<vector<double> > input, 
	vector<int> labels, 
	vector<double> &maleArray, 
	vector<double> &femaleArray
);
void ExtractVAR(
	vector<vector<double> > input, 
	vector<double> maleMeans, 
	vector<double> femaleMeans, 
	vector<int> labels, 
	vector<double> &maleArray, 
	vector<double> &femaleArray
);
//void ExtractAVG(vector<vector<double> > input, double (&array)[30], double (&array)[30]);
//void ExtractVAR(vector<vector<double> > input, vector<double> maleMeans, vector<double> femaleMeans, double (&array)[30], double (&array)[30]);
MatrixXd DiscriminantFunction(MatrixXd x, MatrixXd mu, MatrixXd sigma);

int main(int argc, char *argv[])
{
	DataStorage Data_48x60("48_60");
	DataStorage Data_16x20("16_20");
	//Data_16x20();
	//Data_48x60();
	string inputString;
	do
	{	
		cout << endl
		     << "+============================================================================+\n"
			 << "|Select  0 to obtain 16x20 & 48x60 projected values (fold 1, 2, 3)           |\n"
			 << "|Select  1 to calc 16x20 & 48x60 avg eigen-features (fold 1, 2, 3)           |\n"
			 << "|Select  2 to test data with bayesian classifier                             |\n"
		     << "|Select -1 to exit                                                           |\n"
		     << "+============================================================================+\n"
		     << endl
		     << "Choice: ";

		cin >> inputString;

		if(inputString == "0") 
		{
			cout << "Extracting data from 48x60 files..." << endl;
			Data_48x60.Init(); //Obtain all needed data from files
			cout << "Resizing eigen-feature vectors to 30..." << endl;
			Data_48x60.ReduceSizeTo30();

			cout << "Extracting data from 16x20 files..." << endl;
			Data_16x20.Init(); //Obtain all needed data from files
			cout << "Resizing eigen-feature vectors to 30..." << endl;
			Data_16x20.ReduceSizeTo30();
		}
		else if(inputString == "1")
		{
			cout << "Extracting data from 48x60 files..." << endl;
			Data_48x60.ComputeAvgEigenFeatures(); //Initialize avg eigen-features
			cout << endl;
			
			//cout << "Data_48x60.m_MU_Fold1:" << endl << Data_48x60.m_MU_Fold1 << endl;
			
			cout << "Extracting data from 16x20 files..." << endl;
			Data_16x20.ComputeAvgEigenFeatures(); //Initialize avg eigen-features
		
			//cout << Data_16x20.m_MU_Fold1 << endl;
		
		}
		else if(inputString == "2")
		{
			for (int i = 1; i <= NUM_FOLDS; i++)
			{
				cout << "Classifying fold " << i << " data from 16x20 files..." << endl;
				Data_16x20.BayesianClassifier(i);

				cout << "Classifying fold " << i << " data from 48x60 files..." << endl;
				Data_48x60.BayesianClassifier(i);
			}
		}

	}while(inputString != "-1");
}

DataStorage::DataStorage(string dataSet, string dbRoot) :
	m_DATA_PATH(string(dbRoot + "/" + dataSet + "/"))
{}

void DataStorage::Init()
{
	for (int i = 1; i <= NUM_FOLDS; i++)
	{
		ReadFoldValues(m_TR_PFX, i);
		ReadFoldLabels(m_TR_PFX, i);

		ReadFoldValues(m_VAL_PFX, i);
		ReadFoldLabels(m_VAL_PFX, i);

		ReadFoldValues(m_TS_PFX, i);
		ReadFoldLabels(m_TS_PFX, i);
	}
}

void DataStorage::ReadFoldValues(string setPfx, int foldNum)
{
	ifstream fin;
	string filepath  = m_DATA_PATH + setPfx + "PCA_0" + to_string(foldNum) + ".txt";
	fin.open(filepath.c_str());

	vector<vector<double> > *vecPtr;
	int size;

	if (setPfx == m_TR_PFX)
	{
		size = TR_FOLD_SIZE;
		vecPtr = &(m_TrainVector[foldNum-1]);
	}
	else if (setPfx == m_VAL_PFX)
	{
		size = VAL_FOLD_SIZE;
		vecPtr = &(m_ValidateVector[foldNum-1]);
	}
	else if (setPfx == m_TS_PFX)
	{
		size = TS_FOLD_SIZE;
		vecPtr = &(m_TestVector[foldNum-1]);
	}

	if(!fin.good())
	{
		cout << "error opening file." << filepath << endl;
	}
	else
	{
		double inputDbl;
		string inputString, tempString="";
		for (int j = 0; j < size; ++j)
		{
			(*vecPtr).push_back(vector<double>());
			getline(fin, inputString);
			//cout << "inputString: " << inputString << endl;
			for (int i = 0; i < inputString.size(); ++i)
			{
				if(inputString[i] != ' ')
				{
					tempString += inputString[i];
				}
				else
				{
					double holdDbl = atof(tempString.c_str());
					tempString = "";
					//cout << holdDbl << " ";
					(*vecPtr)[j].push_back(holdDbl);
				}
			}

			//Catch last number
			double holdDbl = atof(tempString.c_str());
			tempString = "";
			(*vecPtr)[j].push_back(holdDbl);

			//cout << holdDbl << " ";

			//cout << endl;
		}
	}
	// cout << setPfx << foldNum << endl;
	// cout << "--------------------------------------" << endl;
	// PrintVector2D(*vecPtr);
	fin.close();
}

void DataStorage::ReadFoldLabels(string setPfx, int foldNum)
{
	ifstream fin;
	string filepath = m_DATA_PATH + m_TG_PFX + setPfx + "PCA_0" + to_string(foldNum) + ".txt";
	fin.open(filepath.c_str());

	vector<int> *vecPtr;
	int size;

	if (setPfx == m_TR_PFX)
	{
		size = TR_FOLD_SIZE;
		vecPtr = &(m_TrainTargetVector[foldNum-1]);
	}
	else if (setPfx == m_VAL_PFX)
	{
		size = VAL_FOLD_SIZE;
		vecPtr = &(m_ValidateTargetVector[foldNum-1]);
	}
	else if (setPfx == m_TS_PFX)
	{
		size = TS_FOLD_SIZE;
		vecPtr = &(m_TestTargetVector[foldNum-1]);
	}

	if(!fin.good())
	{
		cout << "error opening file." << filepath << endl;
	}
	else
	{
		double inputDbl;
		for (int j = 0; j < size; ++j)
		{
			fin >> inputDbl;
			(*vecPtr).push_back(inputDbl);

			//cout << inputDbl << " ";
			//cout << endl;
		}
	}
	// cout << "*vecPtr: " << endl;
	// PrintVector(*vecPtr);
	fin.close();
}

void DataStorage::ReduceSizeTo30()
{
	for (int foldNum = 1; foldNum <= NUM_FOLDS; foldNum++)
	{
		for (int i = 0; i < m_TrainVector[foldNum-1].size(); ++i)
		{
			m_TrainVector[foldNum-1][i].resize(30);
			m_ValidateVector[foldNum-1][i].resize(30);
			m_TestVector[foldNum-1][i].resize(30);
		}
	}

	// //----------------------------------
	// //TRAINING
	// for (int foldNum = 1; foldNum <= NUM_FOLDS; foldNum++)
	// {
	// 	for (int i = 0; i < m_TrainVector[foldNum-1].size(); ++i)
	// 	{
	// 		m_TrainVector[foldNum-1][i].erase(m_TrainVector[foldNum-1][i].begin() + 30, m_TrainVector[foldNum-1][i].end());
	// 	}
	// 	// cout << "m_TrainVector[foldNum-1]: " << endl;
	// 	// PrintVector2D(m_TrainVector[foldNum-1]);
	// }

	// //----------------------------------
	// //VALIDATION

	// for (int foldNum = 1; foldNum <= NUM_FOLDS; foldNum++)
	// {
	// 	for (int i = 0; i < m_ValidateVector[foldNum-1].size(); ++i)
	// 	{
	// 		m_ValidateVector[foldNum-1][i].erase(m_ValidateVector[foldNum-1][i].begin() + 30, m_ValidateVector[foldNum-1][i].end());
	// 	}
	// 	// cout << "m_ValidateVector[foldNum-1]: " << endl;
	// 	// PrintVector2D(m_ValidateVector[foldNum-1]);
	// }
	
	// //----------------------------------
	// //TESTING
	// for (int foldNum = 1; foldNum <= NUM_FOLDS; foldNum++)
	// {
	// 	for (int i = 0; i < m_TestVector[foldNum-1].size(); ++i)
	// 	{
	// 		m_TestVector[foldNum-1][i].erase(m_TestVector[foldNum-1][i].begin() + 30, m_TestVector[foldNum-1][i].end());
	// 	}
	// 	// cout << "m_TestVector[foldNum-1]: " << endl;
	// 	// PrintVector2D(m_TestVector[foldNum-1]);
	// }
}

void DataStorage::ComputeAvgEigenFeatures()
{
	unsigned vec_size = 30;
	vector<double> returnMaleArray(vec_size, 0);
	vector<double> returnFemaleArray(vec_size, 0);

	for (int foldNum = 1; foldNum <= NUM_FOLDS; foldNum++)
	{
		m_Male_MU[foldNum-1] = MatrixXd::Zero(vec_size, 1);
		m_Female_MU[foldNum-1] = MatrixXd::Zero(vec_size, 1);
		m_Male_VAR[foldNum-1] = MatrixXd::Zero(vec_size, vec_size);
		m_Female_VAR[foldNum-1] = MatrixXd::Zero(vec_size, vec_size);

		cout << "Fold " << foldNum << "..." << endl;
		cout << "Computing means..." << endl;

		ExtractAVG(
			m_TrainVector[foldNum-1], 
			m_TrainTargetVector[foldNum-1], 
			returnMaleArray, 
			returnFemaleArray
		);


		for (int i = 0; i < vec_size; ++i)
		{
			m_MeanMaleEigenFeatures[foldNum-1].push_back(returnMaleArray[i]);
			m_MeanFemaleEigenFeatures[foldNum-1].push_back(returnFemaleArray[i]);
		}

		DblVectorToE3Matrix(m_MeanMaleEigenFeatures[foldNum-1], 1, 0, foldNum);
		DblVectorToE3Matrix(m_MeanFemaleEigenFeatures[foldNum-1], 0, 0, foldNum);

		cout << "Computing variances..." << endl;

		// Reset vectors to 0
		fill(returnMaleArray.begin(), returnMaleArray.end(), 0);
		fill(returnFemaleArray.begin(), returnFemaleArray.end(), 0);

		ExtractVAR(
			m_TrainVector[foldNum-1], 
			m_MeanMaleEigenFeatures[foldNum-1], 
			m_MeanFemaleEigenFeatures[foldNum-1], 
			m_TrainTargetVector[foldNum-1], 
			returnMaleArray, 
			returnFemaleArray
		);

		for (int i = 0; i < vec_size; ++i)
		{
			// cout << i << ": " << returnArray[i] << endl;
			m_VarMaleEigenFeatures[foldNum-1].push_back(returnMaleArray[i]);
			m_VarFemaleEigenFeatures[foldNum-1].push_back(returnFemaleArray[i]);
		}

		DblVectorToE3Matrix(m_VarMaleEigenFeatures[foldNum-1], 1, 1, foldNum);
		DblVectorToE3Matrix(m_VarFemaleEigenFeatures[foldNum-1], 0, 1, foldNum);
	}
		
}

void DataStorage::DblVectorToE3Matrix(vector<double> input, bool isMaleVector, bool convertVariance, int fold)
{
	for (int i = 0; i < input.size(); ++i)
	{
		if(!convertVariance)
		{
			if(isMaleVector)
			{
				m_Male_MU[fold-1](i, 0) = input[i];
			}
			else
			{
				m_Female_MU[fold-1](i, 0) = input[i];
			}
		}
		else
		{
			if(isMaleVector)
			{
				m_Male_VAR[fold-1](i, i) = input[i];
			}
			else
			{
				m_Female_VAR[fold-1](i, i) = input[i];
			}
		}
	}
	// cout << endl;
}

void DataStorage::BayesianClassifier(int fold)
{
	int numMales = 0, numFemales = 0;
	cout << "Running testing data..." << endl;

	// PrintVector2D(m_TestVector[fold-1]);

	// cout << "m_Male_MU[" << fold-1 << "]: " << endl << m_Male_MU[fold-1] << endl;
	// cout << "m_Male_VAR[" << fold-1 << "]: " << endl << m_Male_VAR[fold-1] << endl;
	// cout << "m_Female_MU[" << fold-1 << "]: " << endl << m_Female_MU[fold-1] << endl;
	// cout << "m_Female_VAR[" << fold-1 << "]: " << endl << m_Female_VAR[fold-1] << endl;

	for (int j = 0; j < m_TestVector[fold-1].size(); ++j)
	{
		MatrixXd x_vec(m_TestVector[fold-1][j].size(), 1);
		for (int i = 0; i < m_TestVector[fold-1][j].size(); ++i)
		{
			x_vec(i, 0) = m_TestVector[fold-1][j][i];
		}

		// cout << "x_vec: " << endl << x_vec << endl;

		MatrixXd g1 = DiscriminantFunction(x_vec, m_Male_MU[fold-1], m_Male_VAR[fold-1]);
		//cout << "g1: " << g1(0,0) << endl;

		MatrixXd g2 = DiscriminantFunction(x_vec, m_Female_MU[fold-1], m_Female_VAR[fold-1]);
		//cout << "g2: " << g2(0,0) << endl;
		
		double classify = g1(0, 0) - g2(0, 0);
		cout << "(g1-g2): " << classify << endl;
		
		if(classify > 0)
		{
			numMales++;
		}
		else
		{
			numFemales++;
		}
	}
	cout << "Classified Males: " <<	numMales << endl;
	cout << "Classified Females: " << numFemales << endl;
}

MatrixXd DiscriminantFunction(MatrixXd x, MatrixXd mu, MatrixXd sigma)
{
	MatrixXd xt = x.transpose();
	MatrixXd mt = mu.transpose();
	MatrixXd sigma_inv = sigma.inverse();
	MatrixXd W = -0.5 * sigma_inv;
	MatrixXd w = sigma_inv * mu;
	MatrixXd wt  = w.transpose();
	MatrixXd w0 = (-0.5 * mt * sigma_inv * mu);
	w0(0, 0) -= (0.5 * log(sigma.determinant()));

	MatrixXd g_i = (xt * W * x) + (wt * x) + w0;

	return g_i;
}

void ExtractAVG(vector<vector<double> > input, vector<int> labels, vector<double> &maleArray, vector<double> &femaleArray)
{
	int malesInData = 0, femalesInData = 0;
	for (int i = 0; i < labels.size(); ++i)
	{
		if(labels[i] == 1)
		{
			malesInData++;
		}
		else
		{
			femalesInData++;
		}
	}

	cout << "Males in data: " << malesInData << endl;
	cout << "Females in data: " << femalesInData << endl;

	for (int i = 0; i < input.size(); ++i)
	{
		for (int j = 0; j < input[i].size(); ++j)
		{
			if(labels[i] == 1) //male
			{
				maleArray[j] += input[i][j];
			}
			else //female
			{
				femaleArray[j] += input[i][j];
			}
		}
	}

	for (int i = 0; i < input[i].size(); ++i)
	{
		maleArray[i] /= malesInData;
		//cout << "sumDbl: " << array[i] << endl;
	}

	for (int i = 0; i < input[i].size(); ++i)
	{
		femaleArray[i] /= femalesInData;
		//cout << "sumDbl: " << array[i] << endl;
	}
}

void ExtractVAR(
	vector<vector<double> > input, 
	vector<double> maleMeans, 
	vector<double> femaleMeans, 
	vector<int> labels, 
	vector<double> &maleArray, 
	vector<double> &femaleArray
) {
	int malesInData = 0, femalesInData = 0;
	for (int i = 0; i < labels.size(); ++i)
	{
		if(labels[i] == 1)
		{
			malesInData++;
		}
		else
		{
			femalesInData++;
		}
	}

	cout << "Males in data: " << malesInData << endl;
	cout << "Females in data: " << femalesInData << endl;

	for (int i = 0; i < input.size(); ++i)
	{
		for (int j = 0; j < input[i].size(); ++j)
		{
			if(labels[i] == 1) //male
			{
				maleArray[j] += (input[i][j]-maleMeans[j])*(input[i][j]-maleMeans[j]);
			}
			else
			{
				femaleArray[j] += (input[i][j]-femaleMeans[j])*(input[i][j]-femaleMeans[j]);
			}
		}
	}

	for (int i = 0; i < input[i].size(); ++i)
	{
		maleArray[i] /= malesInData;
		femaleArray[i] /= femalesInData;
	}
}

void PrintVector2D(vector<vector<double> > vector)
{
	for (int i = 0; i < vector.size(); ++i)
	{
		for (int j = 0; j < vector[i].size(); ++j)
		{
			cout << vector[i][j] << " ";
		}
		cout << endl;
	}
}

void PrintVector(vector<int> vector)
{
	for (int i = 0; i < vector.size(); ++i)
	{
		cout << vector[i] << " ";
	}
	cout << endl;
}