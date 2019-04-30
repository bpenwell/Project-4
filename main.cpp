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
//Pattern Recognition, Project 3
//April 9, 2019

const int K_VAL = 30;

const int TR_SIZE = 134;
const string trainingFold1_48x60 = "genderdata/48_60/trPCA_01.txt";
const string trainingFold2_48x60 = "genderdata/48_60/trPCA_02.txt";
const string trainingFold3_48x60 = "genderdata/48_60/trPCA_03.txt";
const string targetTrainingFold1_48x60 = "genderdata/48_60/TtrPCA_01.txt";
const string targetTrainingFold2_48x60 = "genderdata/48_60/TtrPCA_02.txt";
const string targetTrainingFold3_48x60 = "genderdata/48_60/TtrPCA_03.txt";
const string trainingFold1_16x20 = "genderdata/16_20/trPCA_01.txt";
const string trainingFold2_16x20 = "genderdata/16_20/trPCA_02.txt";
const string trainingFold3_16x20 = "genderdata/16_20/trPCA_03.txt";
const string targetTrainingFold1_16x20 = "genderdata/16_20/TtrPCA_01.txt";
const string targetTrainingFold2_16x20 = "genderdata/16_20/TtrPCA_02.txt";
const string targetTrainingFold3_16x20 = "genderdata/16_20/TtrPCA_03.txt";

const int VAL_SIZE = 134;
const string validateFold1_48x60 = "genderdata/48_60/valPCA_01.txt";
const string validateFold2_48x60 = "genderdata/48_60/valPCA_02.txt";
const string validateFold3_48x60 = "genderdata/48_60/valPCA_03.txt";
const string targetValidateFold1_48x60 = "genderdata/48_60/TvalPCA_01.txt";
const string targetValidateFold2_48x60 = "genderdata/48_60/TvalPCA_02.txt";
const string targetValidateFold3_48x60 = "genderdata/48_60/TvalPCA_03.txt";
const string validateFold1_16x20 = "genderdata/16_20/valPCA_01.txt";
const string validateFold2_16x20 = "genderdata/16_20/valPCA_02.txt";
const string validateFold3_16x20 = "genderdata/16_20/valPCA_03.txt";
const string targetValidateFold1_16x20 = "genderdata/16_20/TvalPCA_01.txt";
const string targetValidateFold2_16x20 = "genderdata/16_20/TvalPCA_02.txt";
const string targetValidateFold3_16x20 = "genderdata/16_20/TvalPCA_03.txt";

const int TS_SIZE = 133;
const string testFold1_48x60 = "genderdata/48_60/tsPCA_01.txt";
const string testFold2_48x60 = "genderdata/48_60/tsPCA_02.txt";
const string testFold3_48x60 = "genderdata/48_60/tsPCA_03.txt";
const string targetTestFold1_48x60 = "genderdata/48_60/TtsPCA_01.txt";
const string targetTestFold2_48x60 = "genderdata/48_60/TtsPCA_02.txt";
const string targetTestFold3_48x60 = "genderdata/48_60/TtsPCA_03.txt";
const string testFold1_16x20 = "genderdata/16_20/tsPCA_01.txt";
const string testFold2_16x20 = "genderdata/16_20/tsPCA_02.txt";
const string testFold3_16x20 = "genderdata/16_20/tsPCA_03.txt";
const string targetTestFold1_16x20 = "genderdata/16_20/TtsPCA_01.txt";
const string targetTestFold2_16x20 = "genderdata/16_20/TtsPCA_02.txt";
const string targetTestFold3_16x20 = "genderdata/16_20/TtsPCA_03.txt";

class DataStorage_48x60{
public:
	DataStorage_48x60() { 
						  VectorXd temp = VectorXd::Zero(30);
						  m_MU_Fold1 = temp;
						  m_MU_Fold2 = temp;
						  m_MU_Fold3 = temp;
						  m_VAR_Fold1 = temp;
						  m_VAR_Fold2 = temp;
						  m_VAR_Fold3 = temp;
						};
	vector<vector<double> > m_TrainVector_Fold1;
	vector<vector<double> > m_TrainVector_Fold2;
	vector<vector<double> > m_TrainVector_Fold3;
	vector<int> m_TrainTargetVector_Fold1;
	vector<int> m_TrainTargetVector_Fold2;
	vector<int> m_TrainTargetVector_Fold3;

	vector<vector<double> > m_ValidateVector_Fold1;
	vector<vector<double> > m_ValidateVector_Fold2;
	vector<vector<double> > m_ValidateVector_Fold3;
	vector<int> m_ValidateTargetVector_Fold1;
	vector<int> m_ValidateTargetVector_Fold2;
	vector<int> m_ValidateTargetVector_Fold3;

	vector<vector<double> > m_TestVector_Fold1;
	vector<vector<double> > m_TestVector_Fold2;
	vector<vector<double> > m_TestVector_Fold3;
	vector<int> m_TestTargetVector_Fold1;
	vector<int> m_TestTargetVector_Fold2;
	vector<int> m_TestTargetVector_Fold3;

	vector<double> m_MeanEigenFeatures_Fold1;
	vector<double> m_MeanEigenFeatures_Fold2;
	vector<double> m_MeanEigenFeatures_Fold3;
	VectorXd m_MU_Fold1;
	VectorXd m_MU_Fold2;
	VectorXd m_MU_Fold3;

	vector<double> m_VarEigenFeatures_Fold1;
	vector<double> m_VarEigenFeatures_Fold2;
	vector<double> m_VarEigenFeatures_Fold3;
	VectorXd m_VAR_Fold1;
	VectorXd m_VAR_Fold2;
	VectorXd m_VAR_Fold3;


	void Init();
	void ReduceSizeTo30();
	void ComputeAvgEigenFeatures();
	void DblVectorToE3Vector(vector<double> input, bool convertVariance, int fold);
};

class DataStorage_16x20{
public:
	DataStorage_16x20() { 
						  VectorXd temp = VectorXd::Zero(30);
						  m_MU_Fold1 = temp;
						  m_MU_Fold2 = temp;
						  m_MU_Fold3 = temp;
						  m_VAR_Fold1 = temp;
						  m_VAR_Fold2 = temp;
						  m_VAR_Fold3 = temp;
						};
	vector<vector<double> > m_TrainVector_Fold1;
	vector<vector<double> > m_TrainVector_Fold2;
	vector<vector<double> > m_TrainVector_Fold3;
	vector<int> m_TrainTargetVector_Fold1;
	vector<int> m_TrainTargetVector_Fold2;
	vector<int> m_TrainTargetVector_Fold3;

	vector<vector<double> > m_ValidateVector_Fold1;
	vector<vector<double> > m_ValidateVector_Fold2;
	vector<vector<double> > m_ValidateVector_Fold3;
	vector<int> m_ValidateTargetVector_Fold1;
	vector<int> m_ValidateTargetVector_Fold2;
	vector<int> m_ValidateTargetVector_Fold3;

	vector<vector<double> > m_TestVector_Fold1;
	vector<vector<double> > m_TestVector_Fold2;
	vector<vector<double> > m_TestVector_Fold3;
	vector<int> m_TestTargetVector_Fold1;
	vector<int> m_TestTargetVector_Fold2;
	vector<int> m_TestTargetVector_Fold3;

	vector<double> m_MeanEigenFeatures_Fold1;
	vector<double> m_MeanEigenFeatures_Fold2;
	vector<double> m_MeanEigenFeatures_Fold3;
	VectorXd m_MU_Fold1;
	VectorXd m_MU_Fold2;
	VectorXd m_MU_Fold3;

	vector<double> m_VarEigenFeatures_Fold1;
	vector<double> m_VarEigenFeatures_Fold2;
	vector<double> m_VarEigenFeatures_Fold3;
	VectorXd m_VAR_Fold1;
	VectorXd m_VAR_Fold2;
	VectorXd m_VAR_Fold3;

	void Init();
	void ReduceSizeTo30();
	void ComputeAvgEigenFeatures();
	void DblVectorToE3Vector(vector<double> input, bool convertVariance, int fold);
};

void PrintVector2D(vector<vector<double> > vector);
void PrintVector(vector<int> vector);
void ExtractAVG(vector<vector<double> > input, double (&array)[30]);
void ExtractVAR(vector<vector<double> > input, vector<double> means, double (&array)[30]);

int main()
{
	DataStorage_48x60 Data_48x60;
	DataStorage_16x20 Data_16x20;
	//Data_16x20();
	//Data_48x60();
	string inputString;
	do
	{	
		cout << endl
		     << "+============================================================================+\n"
			 << "|Select  0 to obtain 16x20 & 48x60 projected values (fold 1, 2, 3)           |\n"
			 << "|Select  1 to calc 16x20 & 48x60 avg eigen-features (fold 1, 2, 3)           |\n"
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
			
			cout << "Extracting data from 16x20 files..." << endl;
			Data_16x20.ComputeAvgEigenFeatures(); //Initialize avg eigen-features
		}

	}while(inputString != "-1");
}

void DataStorage_48x60::Init()
{
	ifstream fin;

	//------------------------------------
	//TRAINING FOLD 1
	if(1)
	{
		fin.open(trainingFold1_48x60.c_str());
		if(!fin.good())
		{
			cout << "error opening file." << trainingFold1_48x60 << endl;
		}
		else
		{
			double inputDbl;
			string inputString, tempString="";
			for (int j = 0; j < TR_SIZE; ++j)
			{
				m_TrainVector_Fold1.push_back(vector<double>());
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
						m_TrainVector_Fold1[j].push_back(holdDbl);
					}
				}

				//Catch last number
				double holdDbl = atof(tempString.c_str());
				tempString = "";
				m_TrainVector_Fold1[j].push_back(holdDbl);

				//cout << holdDbl << " ";

				//cout << endl;
			}
		}
		//cout << "m_TrainVector_Fold1: " << endl;
		//PrintVector2D(m_TrainVector_Fold1);
		fin.close();

		//TRAINING FOLD 1 LABELS
		fin.open(targetTrainingFold1_48x60.c_str());
		if(!fin.good())
		{
			cout << "error opening file." << targetTrainingFold1_48x60 << endl;
		}
		else
		{
			double inputDbl;
			for (int j = 0; j < TR_SIZE; ++j)
			{
				fin >> inputDbl;
				m_TrainTargetVector_Fold1.push_back(inputDbl);

				//cout << inputDbl << " ";
				//cout << endl;
			}
		}
		// cout << "m_TrainTargetVector_Fold1: " << endl;
		// PrintVector(m_TrainTargetVector_Fold1);
		fin.close();
	}
	//------------------------------------

	//TRAINING FOLD 2
	if(1)
	{
		fin.open(trainingFold2_48x60.c_str());
		if(!fin.good())
		{
			cout << "error opening " << trainingFold2_48x60 << endl;
		}
		else
		{
			double inputDbl;
			string inputString, tempString="";
			for (int j = 0; j < TR_SIZE; ++j)
			{
				m_TrainVector_Fold2.push_back(vector<double>());
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
						m_TrainVector_Fold2[j].push_back(holdDbl);
					}
				}

				//Catch last number
				double holdDbl = atof(tempString.c_str());
				tempString = "";
				m_TrainVector_Fold2[j].push_back(holdDbl);

				//cout << holdDbl << " ";

				//cout << endl;
			}
		}
		//cout << "m_TrainVector_Fold2: " << endl;
		//PrintVector2D(m_TrainVector_Fold2);
		fin.close();

		//TRAINING FOLD 2 LABELS
		fin.open(targetTrainingFold2_48x60.c_str());
		if(!fin.good())
		{
			cout << "error opening file." << targetTrainingFold2_48x60 << endl;
		}
		else
		{
			double inputDbl;
			for (int j = 0; j < TR_SIZE; ++j)
			{
				fin >> inputDbl;
				m_TrainTargetVector_Fold2.push_back(inputDbl);

				//cout << inputDbl << " ";
				//cout << endl;
			}
		}
		// cout << "m_TrainTargetVector_Fold2: " << endl;
		// PrintVector(m_TrainTargetVector_Fold2);
		fin.close();
	}
	//------------------------------------

	//TRAINING FOLD 3
	if(1)
	{

		fin.open(trainingFold3_48x60.c_str());
		if(!fin.good())
		{
			cout << "error opening " << trainingFold3_48x60 << endl;
		}
		else
		{
			double inputDbl;
			string inputString, tempString="";
			for (int j = 0; j < TR_SIZE; ++j)
			{
				m_TrainVector_Fold3.push_back(vector<double>());
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
						m_TrainVector_Fold3[j].push_back(holdDbl);
					}
				}

				//Catch last number
				double holdDbl = atof(tempString.c_str());
				tempString = "";
				m_TrainVector_Fold3[j].push_back(holdDbl);

				//cout << holdDbl << " ";

				//cout << endl;
			}
		}
		//cout << "m_TrainVector_Fold3: " << endl;
		//PrintVector2D(m_TrainVector_Fold3);
		fin.close();

		//TRAINING FOLD 3 LABELS
		fin.open(targetTrainingFold3_48x60.c_str());
		if(!fin.good())
		{
			cout << "error opening file." << targetTrainingFold3_48x60 << endl;
		}
		else
		{
			double inputDbl;
			for (int j = 0; j < TR_SIZE; ++j)
			{
				fin >> inputDbl;
				m_TrainTargetVector_Fold3.push_back(inputDbl);

				//cout << inputDbl << " ";
				//cout << endl;
			}
		}
		// cout << "m_TrainTargetVector_Fold3:" << endl;
		// PrintVector(m_TrainTargetVector_Fold3);
		fin.close();
	}
	//------------------------------------

	//------------------------------------
	//VALIDATE FOLD 1
	if(1)
	{
		fin.open(validateFold1_48x60.c_str());
		if(!fin.good())
		{
			cout << "error opening file." << validateFold1_48x60 << endl;
		}
		else
		{
			double inputDbl;
			string inputString, tempString="";
			for (int j = 0; j < VAL_SIZE; ++j)
			{
				m_ValidateVector_Fold1.push_back(vector<double>());
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
						m_ValidateVector_Fold1[j].push_back(holdDbl);
					}
				}

				//Catch last number
				double holdDbl = atof(tempString.c_str());
				tempString = "";
				m_ValidateVector_Fold1[j].push_back(holdDbl);

				//cout << holdDbl << " ";

				//cout << endl;
			}
		}
		//cout << "m_ValidateVector_Fold1: " << endl;
		//PrintVector2D(m_ValidateVector_Fold1);
		fin.close();

		//VALIDATE FOLD 1 LABELS
		fin.open(targetValidateFold1_48x60.c_str());
		if(!fin.good())
		{
			cout << "error opening file." << targetValidateFold1_48x60 << endl;
		}
		else
		{
			double inputDbl;
			for (int j = 0; j < VAL_SIZE-1; ++j)
			{
				fin >> inputDbl;
				m_ValidateTargetVector_Fold1.push_back(inputDbl);

				//cout << inputDbl << " ";
				//cout << endl;
			}
		}
		//cout << "m_ValidateTargetVector_Fold1:" << endl;
		//PrintVector(m_ValidateTargetVector_Fold1);
		fin.close();
	}
	//------------------------------------

	//VALIDATE FOLD 2
	if(1)
	{
		fin.open(validateFold2_48x60.c_str());
		if(!fin.good())
		{
			cout << "error opening " << validateFold2_48x60 << endl;
		}
		else
		{
			double inputDbl;
			string inputString, tempString="";
			for (int j = 0; j < VAL_SIZE; ++j)
			{
				m_ValidateVector_Fold2.push_back(vector<double>());
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
						m_ValidateVector_Fold2[j].push_back(holdDbl);
					}
				}

				//Catch last number
				double holdDbl = atof(tempString.c_str());
				tempString = "";
				m_ValidateVector_Fold2[j].push_back(holdDbl);

				//cout << holdDbl << " ";

				//cout << endl;
			}
		}
		//cout << "m_ValidateVector_Fold2: " << endl;
		//PrintVector2D(m_ValidateVector_Fold2);
		fin.close();

		//VALIDATE FOLD 2 LABELS
		fin.open(targetValidateFold2_48x60.c_str());
		if(!fin.good())
		{
			cout << "error opening file." << targetValidateFold2_48x60 << endl;
		}
		else
		{
			double inputDbl;
			for (int j = 0; j < VAL_SIZE; ++j)
			{
				fin >> inputDbl;
				m_ValidateTargetVector_Fold2.push_back(inputDbl);

				//cout << inputDbl << " ";
				//cout << endl;
			}
		}
		//cout << "m_ValidateTargetVector_Fold2:" << endl;
		//PrintVector(m_ValidateTargetVector_Fold2);
		fin.close();
	}
	//------------------------------------
	
	//VALIDATE FOLD 3
	if(1)
	{

		fin.open(validateFold3_48x60.c_str());
		if(!fin.good())
		{
			cout << "error opening " << validateFold3_48x60 << endl;
		}
		else
		{
			double inputDbl;
			string inputString, tempString="";
			for (int j = 0; j < VAL_SIZE; ++j)
			{
				m_ValidateVector_Fold3.push_back(vector<double>());
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
						m_ValidateVector_Fold3[j].push_back(holdDbl);
					}
				}

				//Catch last number
				double holdDbl = atof(tempString.c_str());
				tempString = "";
				m_ValidateVector_Fold3[j].push_back(holdDbl);

				//cout << holdDbl << " ";

				//cout << endl;
			}
		}
		//cout << "m_ValidateVector_Fold3:" << endl;
		//PrintVector2D(m_ValidateVector_Fold3);
		fin.close();

		//VALIDATE FOLD 3 LABELS
		fin.open(targetValidateFold3_48x60.c_str());
		if(!fin.good())
		{
			cout << "error opening file." << targetValidateFold3_48x60 << endl;
		}
		else
		{
			double inputDbl;
			for (int j = 0; j < VAL_SIZE; ++j)
			{
				fin >> inputDbl;
				m_ValidateTargetVector_Fold3.push_back(inputDbl);

				//cout << inputDbl << " ";
				//cout << endl;
			}
		}
		//cout << "m_ValidateTargetVector_Fold3:" << endl;
		//PrintVector(m_ValidateTargetVector_Fold3);
		fin.close();
	}
	//------------------------------------

	//------------------------------------
	//TEST FOLD 1
	if(1)
	{
		fin.open(testFold1_48x60.c_str());
		if(!fin.good())
		{
			cout << "error opening file." << testFold1_48x60 << endl;
		}
		else
		{
			double inputDbl;
			string inputString, tempString="";
			for (int j = 0; j < TS_SIZE+1; ++j)
			{
				m_TestVector_Fold1.push_back(vector<double>());
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
						m_TestVector_Fold1[j].push_back(holdDbl);
					}
				}

				//Catch last number
				double holdDbl = atof(tempString.c_str());
				tempString = "";
				m_TestVector_Fold1[j].push_back(holdDbl);

				//cout << holdDbl << " ";

				//cout << endl;
			}
		}
		//cout << "m_TestVector_Fold1: " << endl;
		//PrintVector2D(m_TestVector_Fold1);
		fin.close();

		//TEST FOLD 1 LABELS
		fin.open(targetTestFold1_48x60.c_str());
		if(!fin.good())
		{
			cout << "error opening file." << targetTestFold1_48x60 << endl;
		}
		else
		{
			double inputDbl;
			for (int j = 0; j < TS_SIZE; ++j)
			{
				fin >> inputDbl;
				m_TestTargetVector_Fold1.push_back(inputDbl);

				//cout << inputDbl << " ";
				//cout << endl;
			}
		}
		// cout << "m_TestTargetVector_Fold1:" << endl;
		// PrintVector(m_TestTargetVector_Fold1);
		fin.close();
	}
	//------------------------------------

	//TEST FOLD 2
	if(1)
	{
		fin.open(testFold2_48x60.c_str());
		if(!fin.good())
		{
			cout << "error opening " << testFold2_48x60 << endl;
		}
		else
		{
			double inputDbl;
			string inputString, tempString="";
			for (int j = 0; j < TS_SIZE+1; ++j)
			{
				m_TestVector_Fold2.push_back(vector<double>());
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
						m_TestVector_Fold2[j].push_back(holdDbl);
					}
				}

				//Catch last number
				double holdDbl = atof(tempString.c_str());
				tempString = "";
				m_TestVector_Fold2[j].push_back(holdDbl);

				//cout << holdDbl << " ";

				//cout << endl;
			}
		}
		//cout << "m_TestVector_Fold2: " << endl;
		//PrintVector2D(m_TestVector_Fold2);
		fin.close();

		//TEST FOLD 2 LABELS
		fin.open(targetTestFold2_48x60.c_str());
		if(!fin.good())
		{
			cout << "error opening file." << targetTestFold2_48x60 << endl;
		}
		else
		{
			double inputDbl;
			for (int j = 0; j < TS_SIZE; ++j)
			{
				fin >> inputDbl;
				m_TestTargetVector_Fold2.push_back(inputDbl);

				//cout << inputDbl << " ";
				//cout << endl;
			}
		}
		// cout << "m_TestTargetVector_Fold2:" << endl;
		// PrintVector(m_TestTargetVector_Fold2);
		fin.close();
	}
	//------------------------------------
	
	//TEST FOLD 3
	if(1)
	{

		fin.open(testFold3_48x60.c_str());
		if(!fin.good())
		{
			cout << "error opening " << testFold3_48x60 << endl;
		}
		else
		{
			double inputDbl;
			string inputString, tempString="";
			for (int j = 0; j < TS_SIZE+1; ++j)
			{
				m_TestVector_Fold3.push_back(vector<double>());
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
						m_TestVector_Fold3[j].push_back(holdDbl);
					}
				}

				//Catch last number
				double holdDbl = atof(tempString.c_str());
				tempString = "";
				m_TestVector_Fold3[j].push_back(holdDbl);

				//cout << holdDbl << " ";

				//cout << endl;
			}
		}
		//cout << "m_TestVector_Fold3:" << endl;
		//PrintVector2D(m_TestVector_Fold3);
		fin.close();

		//TEST FOLD 3 LABELS
		fin.open(targetTestFold3_48x60.c_str());
		if(!fin.good())
		{
			cout << "error opening file." << targetTestFold3_48x60 << endl;
		}
		else
		{
			double inputDbl;
			for (int j = 0; j < TS_SIZE; ++j)
			{
				fin >> inputDbl;
				m_TestTargetVector_Fold3.push_back(inputDbl);

				//cout << inputDbl << " ";
				//cout << endl;
			}
		}
		// cout << "m_TestTargetVector_Fold3:" << endl;
		// PrintVector(m_TestTargetVector_Fold3);
		fin.close();
	}
	//------------------------------------
}

void DataStorage_48x60::ReduceSizeTo30()
{
	//----------------------------------
	//TRAINING
	for (int i = 0; i < m_TrainVector_Fold1.size(); ++i)
	{
		m_TrainVector_Fold1[i].erase(m_TrainVector_Fold1[i].begin() + 30, m_TrainVector_Fold1[i].end());
	}
	// cout << "m_TrainVector_Fold1: " << endl;
	// PrintVector2D(m_TrainVector_Fold1);
	for (int i = 0; i < m_TrainVector_Fold2.size(); ++i)
	{
		m_TrainVector_Fold2[i].erase(m_TrainVector_Fold2[i].begin() + 30, m_TrainVector_Fold2[i].end());
	}
	// cout << "m_TrainVector_Fold2: " << endl;
	// PrintVector2D(m_TrainVector_Fold2);
	for (int i = 0; i < m_TrainVector_Fold3.size(); ++i)
	{
		m_TrainVector_Fold3[i].erase(m_TrainVector_Fold3[i].begin() + 30, m_TrainVector_Fold3[i].end());
	}
	// cout << "m_TrainVector_Fold3: " << endl;
	// PrintVector2D(m_TrainVector_Fold3);

	//----------------------------------
	//VALIDATION
	for (int i = 0; i < m_ValidateVector_Fold1.size(); ++i)
	{
		m_ValidateVector_Fold1[i].erase(m_ValidateVector_Fold1[i].begin() + 30, m_ValidateVector_Fold1[i].end());
	}
	// cout << "m_ValidateVector_Fold1: " << endl;
	// PrintVector2D(m_ValidateVector_Fold1);
	
	for (int i = 0; i < m_ValidateVector_Fold2.size(); ++i)
	{
		m_ValidateVector_Fold2[i].erase(m_ValidateVector_Fold2[i].begin() + 30, m_ValidateVector_Fold2[i].end());
	}
	// cout << "m_ValidateVector_Fold2: " << endl;
	// PrintVector2D(m_ValidateVector_Fold2);

	for (int i = 0; i < m_ValidateVector_Fold3.size(); ++i)
	{
		m_ValidateVector_Fold3[i].erase(m_ValidateVector_Fold3[i].begin() + 30, m_ValidateVector_Fold3[i].end());
	}
	// cout << "m_ValidateVector_Fold3: " << endl;
	// PrintVector2D(m_ValidateVector_Fold3);
	
	//----------------------------------
	//TESTING
	for (int i = 0; i < m_TestVector_Fold1.size(); ++i)
	{
		m_TestVector_Fold1[i].erase(m_TestVector_Fold1[i].begin() + 30, m_TestVector_Fold1[i].end());
	}
	// cout << "m_TestVector_Fold1: " << endl;
	// PrintVector2D(m_TestVector_Fold1);
	
	for (int i = 0; i < m_TestVector_Fold2.size(); ++i)
	{
		m_TestVector_Fold2[i].erase(m_TestVector_Fold2[i].begin() + 30, m_TestVector_Fold2[i].end());
	}
	// cout << "m_TestVector_Fold2: " << endl;
	// PrintVector2D(m_TestVector_Fold2);

	for (int i = 0; i < m_TestVector_Fold3.size(); ++i)
	{
		m_TestVector_Fold3[i].erase(m_TestVector_Fold3[i].begin() + 30, m_TestVector_Fold3[i].end());
	}
	// cout << "m_TestVector_Fold3: " << endl;
	// PrintVector2D(m_TestVector_Fold3);
}

void DataStorage_16x20::Init()
{
	ifstream fin;

	//------------------------------------
	//TRAINING FOLD 1
	if(1)
	{
		fin.open(trainingFold1_16x20.c_str());
		if(!fin.good())
		{
			cout << "error opening file." << trainingFold1_16x20 << endl;
		}
		else
		{
			double inputDbl;
			string inputString, tempString="";
			for (int j = 0; j < TR_SIZE; ++j)
			{
				m_TrainVector_Fold1.push_back(vector<double>());
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
						m_TrainVector_Fold1[j].push_back(holdDbl);
					}
				}

				//Catch last number
				double holdDbl = atof(tempString.c_str());
				tempString = "";
				m_TrainVector_Fold1[j].push_back(holdDbl);

				//cout << holdDbl << " ";

				//cout << endl;
			}
		}
		//cout << "m_TrainVector_Fold1: " << endl;
		//PrintVector2D(m_TrainVector_Fold1);
		fin.close();

		//TRAINING FOLD 1 LABELS
		fin.open(targetTrainingFold1_16x20.c_str());
		if(!fin.good())
		{
			cout << "error opening file." << targetTrainingFold1_16x20 << endl;
		}
		else
		{
			double inputDbl;
			for (int j = 0; j < TR_SIZE; ++j)
			{
				fin >> inputDbl;
				m_TrainTargetVector_Fold1.push_back(inputDbl);

				//cout << inputDbl << " ";
				//cout << endl;
			}
		}
		// cout << "m_TrainTargetVector_Fold1: " << endl;
		// PrintVector(m_TrainTargetVector_Fold1);
		fin.close();
	}
	//------------------------------------

	//TRAINING FOLD 2
	if(1)
	{
		fin.open(trainingFold2_16x20.c_str());
		if(!fin.good())
		{
			cout << "error opening " << trainingFold2_16x20 << endl;
		}
		else
		{
			double inputDbl;
			string inputString, tempString="";
			for (int j = 0; j < TR_SIZE; ++j)
			{
				m_TrainVector_Fold2.push_back(vector<double>());
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
						m_TrainVector_Fold2[j].push_back(holdDbl);
					}
				}

				//Catch last number
				double holdDbl = atof(tempString.c_str());
				tempString = "";
				m_TrainVector_Fold2[j].push_back(holdDbl);

				//cout << holdDbl << " ";

				//cout << endl;
			}
		}
		//cout << "m_TrainVector_Fold2: " << endl;
		//PrintVector2D(m_TrainVector_Fold2);
		fin.close();

		//TRAINING FOLD 2 LABELS
		fin.open(targetTrainingFold2_16x20.c_str());
		if(!fin.good())
		{
			cout << "error opening file." << targetTrainingFold2_16x20 << endl;
		}
		else
		{
			double inputDbl;
			for (int j = 0; j < TR_SIZE; ++j)
			{
				fin >> inputDbl;
				m_TrainTargetVector_Fold2.push_back(inputDbl);

				//cout << inputDbl << " ";
				//cout << endl;
			}
		}
		// cout << "m_TrainTargetVector_Fold2: " << endl;
		// PrintVector(m_TrainTargetVector_Fold2);
		fin.close();
	}
	//------------------------------------

	//TRAINING FOLD 3
	if(1)
	{

		fin.open(trainingFold3_16x20.c_str());
		if(!fin.good())
		{
			cout << "error opening " << trainingFold3_16x20 << endl;
		}
		else
		{
			double inputDbl;
			string inputString, tempString="";
			for (int j = 0; j < TR_SIZE; ++j)
			{
				m_TrainVector_Fold3.push_back(vector<double>());
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
						m_TrainVector_Fold3[j].push_back(holdDbl);
					}
				}

				//Catch last number
				double holdDbl = atof(tempString.c_str());
				tempString = "";
				m_TrainVector_Fold3[j].push_back(holdDbl);

				//cout << holdDbl << " ";

				//cout << endl;
			}
		}
		//cout << "m_TrainVector_Fold3: " << endl;
		//PrintVector2D(m_TrainVector_Fold3);
		fin.close();

		//TRAINING FOLD 3 LABELS
		fin.open(targetTrainingFold3_16x20.c_str());
		if(!fin.good())
		{
			cout << "error opening file." << targetTrainingFold3_16x20 << endl;
		}
		else
		{
			double inputDbl;
			for (int j = 0; j < TR_SIZE; ++j)
			{
				fin >> inputDbl;
				m_TrainTargetVector_Fold3.push_back(inputDbl);

				//cout << inputDbl << " ";
				//cout << endl;
			}
		}
		// cout << "m_TrainTargetVector_Fold3:" << endl;
		// PrintVector(m_TrainTargetVector_Fold3);
		fin.close();
	}
	//------------------------------------

	//------------------------------------
	//VALIDATE FOLD 1
	if(1)
	{
		fin.open(validateFold1_16x20.c_str());
		if(!fin.good())
		{
			cout << "error opening file." << validateFold1_16x20 << endl;
		}
		else
		{
			double inputDbl;
			string inputString, tempString="";
			for (int j = 0; j < VAL_SIZE; ++j)
			{
				m_ValidateVector_Fold1.push_back(vector<double>());
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
						m_ValidateVector_Fold1[j].push_back(holdDbl);
					}
				}

				//Catch last number
				double holdDbl = atof(tempString.c_str());
				tempString = "";
				m_ValidateVector_Fold1[j].push_back(holdDbl);

				//cout << holdDbl << " ";

				//cout << endl;
			}
		}
		//cout << "m_ValidateVector_Fold1: " << endl;
		//PrintVector2D(m_ValidateVector_Fold1);
		fin.close();

		//VALIDATE FOLD 1 LABELS
		fin.open(targetValidateFold1_16x20.c_str());
		if(!fin.good())
		{
			cout << "error opening file." << targetValidateFold1_16x20 << endl;
		}
		else
		{
			double inputDbl;
			for (int j = 0; j < VAL_SIZE-1; ++j)
			{
				fin >> inputDbl;
				m_ValidateTargetVector_Fold1.push_back(inputDbl);

				//cout << inputDbl << " ";
				//cout << endl;
			}
		}
		// cout << "m_ValidateTargetVector_Fold1:" << endl;
		// PrintVector(m_ValidateTargetVector_Fold1);
		fin.close();
	}
	//------------------------------------

	//VALIDATE FOLD 2
	if(1)
	{
		fin.open(validateFold2_16x20.c_str());
		if(!fin.good())
		{
			cout << "error opening " << validateFold2_16x20 << endl;
		}
		else
		{
			double inputDbl;
			string inputString, tempString="";
			for (int j = 0; j < VAL_SIZE; ++j)
			{
				m_ValidateVector_Fold2.push_back(vector<double>());
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
						m_ValidateVector_Fold2[j].push_back(holdDbl);
					}
				}

				//Catch last number
				double holdDbl = atof(tempString.c_str());
				tempString = "";
				m_ValidateVector_Fold2[j].push_back(holdDbl);

				//cout << holdDbl << " ";

				//cout << endl;
			}
		}
		//cout << "m_ValidateVector_Fold2: " << endl;
		//PrintVector2D(m_ValidateVector_Fold2);
		fin.close();

		//VALIDATE FOLD 2 LABELS
		fin.open(targetValidateFold2_16x20.c_str());
		if(!fin.good())
		{
			cout << "error opening file." << targetValidateFold2_16x20 << endl;
		}
		else
		{
			double inputDbl;
			for (int j = 0; j < VAL_SIZE-1; ++j)
			{
				fin >> inputDbl;
				m_ValidateTargetVector_Fold2.push_back(inputDbl);

				//cout << inputDbl << " ";
				//cout << endl;
			}
		}
		//cout << "m_ValidateTargetVector_Fold2:" << endl;
		//PrintVector(m_ValidateTargetVector_Fold2);
		fin.close();
	}
	//------------------------------------
	
	//VALIDATE FOLD 3
	if(1)
	{

		fin.open(validateFold3_16x20.c_str());
		if(!fin.good())
		{
			cout << "error opening " << validateFold3_16x20 << endl;
		}
		else
		{
			double inputDbl;
			string inputString, tempString="";
			for (int j = 0; j < VAL_SIZE; ++j)
			{
				m_ValidateVector_Fold3.push_back(vector<double>());
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
						m_ValidateVector_Fold3[j].push_back(holdDbl);
					}
				}

				//Catch last number
				double holdDbl = atof(tempString.c_str());
				tempString = "";
				m_ValidateVector_Fold3[j].push_back(holdDbl);

				//cout << holdDbl << " ";

				//cout << endl;
			}
		}
		//cout << "m_ValidateVector_Fold3:" << endl;
		//PrintVector2D(m_ValidateVector_Fold3);
		fin.close();

		//VALIDATE FOLD 3 LABELS
		fin.open(targetValidateFold3_16x20.c_str());
		if(!fin.good())
		{
			cout << "error opening file." << targetValidateFold3_16x20 << endl;
		}
		else
		{
			double inputDbl;
			for (int j = 0; j < VAL_SIZE-1; ++j)
			{
				fin >> inputDbl;
				m_ValidateTargetVector_Fold3.push_back(inputDbl);

				//cout << inputDbl << " ";
				//cout << endl;
			}
		}
		// cout << "m_ValidateTargetVector_Fold3:" << endl;
		// PrintVector(m_ValidateTargetVector_Fold3);
		fin.close();
	}
	//------------------------------------

	//------------------------------------
	//TEST FOLD 1
	if(1)
	{
		fin.open(testFold1_16x20.c_str());
		if(!fin.good())
		{
			cout << "error opening file." << testFold1_16x20 << endl;
		}
		else
		{
			double inputDbl;
			string inputString, tempString="";
			for (int j = 0; j < TS_SIZE+1; ++j)
			{
				m_TestVector_Fold1.push_back(vector<double>());
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
						m_TestVector_Fold1[j].push_back(holdDbl);
					}
				}

				//Catch last number
				double holdDbl = atof(tempString.c_str());
				tempString = "";
				m_TestVector_Fold1[j].push_back(holdDbl);

				//cout << holdDbl << " ";

				//cout << endl;
			}
		}
		//cout << "m_TestVector_Fold1: " << endl;
		//PrintVector2D(m_TestVector_Fold1);
		fin.close();

		//TEST FOLD 1 LABELS
		fin.open(targetTestFold1_16x20.c_str());
		if(!fin.good())
		{
			cout << "error opening file." << targetTestFold1_16x20 << endl;
		}
		else
		{
			double inputDbl;
			for (int j = 0; j < TS_SIZE; ++j)
			{
				fin >> inputDbl;
				m_TestTargetVector_Fold1.push_back(inputDbl);

				//cout << inputDbl << " ";
				//cout << endl;
			}
		}
		// cout << "m_TestTargetVector_Fold1:" << endl;
		// PrintVector(m_TestTargetVector_Fold1);
		fin.close();
	}
	//------------------------------------

	//TEST FOLD 2
	if(1)
	{
		fin.open(testFold2_16x20.c_str());
		if(!fin.good())
		{
			cout << "error opening " << testFold2_16x20 << endl;
		}
		else
		{
			double inputDbl;
			string inputString, tempString="";
			for (int j = 0; j < TS_SIZE+1; ++j)
			{
				m_TestVector_Fold2.push_back(vector<double>());
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
						m_TestVector_Fold2[j].push_back(holdDbl);
					}
				}

				//Catch last number
				double holdDbl = atof(tempString.c_str());
				tempString = "";
				m_TestVector_Fold2[j].push_back(holdDbl);

				//cout << holdDbl << " ";

				//cout << endl;
			}
		}
		//cout << "m_TestVector_Fold2: " << endl;
		//PrintVector2D(m_TestVector_Fold2);
		fin.close();

		//TEST FOLD 2 LABELS
		fin.open(targetTestFold2_16x20.c_str());
		if(!fin.good())
		{
			cout << "error opening file." << targetTestFold2_16x20 << endl;
		}
		else
		{
			double inputDbl;
			for (int j = 0; j < TS_SIZE; ++j)
			{
				fin >> inputDbl;
				m_TestTargetVector_Fold2.push_back(inputDbl);

				//cout << inputDbl << " ";
				//cout << endl;
			}
		}
		// cout << "m_TestTargetVector_Fold2:" << endl;
		// PrintVector(m_TestTargetVector_Fold2);
		fin.close();
	}
	//------------------------------------
	
	//TEST FOLD 3
	if(1)
	{

		fin.open(testFold3_16x20.c_str());
		if(!fin.good())
		{
			cout << "error opening " << testFold3_16x20 << endl;
		}
		else
		{
			double inputDbl;
			string inputString, tempString="";
			for (int j = 0; j < TS_SIZE+1; ++j)
			{
				m_TestVector_Fold3.push_back(vector<double>());
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
						m_TestVector_Fold3[j].push_back(holdDbl);
					}
				}

				//Catch last number
				double holdDbl = atof(tempString.c_str());
				tempString = "";
				m_TestVector_Fold3[j].push_back(holdDbl);

				//cout << holdDbl << " ";

				//cout << endl;
			}
		}
		//cout << "m_TestVector_Fold3:" << endl;
		//PrintVector2D(m_TestVector_Fold3);
		fin.close();

		//TEST FOLD 3 LABELS
		fin.open(targetTestFold3_16x20.c_str());
		if(!fin.good())
		{
			cout << "error opening file." << targetTestFold3_16x20 << endl;
		}
		else
		{
			double inputDbl;
			for (int j = 0; j < TS_SIZE; ++j)
			{
				fin >> inputDbl;
				m_TestTargetVector_Fold3.push_back(inputDbl);

				//cout << inputDbl << " ";
				//cout << endl;
			}
		}
		// cout << "m_TestTargetVector_Fold3:" << endl;
		// PrintVector(m_TestTargetVector_Fold3);
		fin.close();
	}
	//------------------------------------
}

void DataStorage_16x20::ReduceSizeTo30()
{
	//----------------------------------
	//TRAINING
	for (int i = 0; i < m_TrainVector_Fold1.size(); ++i)
	{
		m_TrainVector_Fold1[i].erase(m_TrainVector_Fold1[i].begin() + 30, m_TrainVector_Fold1[i].end());
	}
	//cout << "m_TrainVector_Fold1: " << endl;
	//PrintVector2D(m_TrainVector_Fold1);
	for (int i = 0; i < m_TrainVector_Fold2.size(); ++i)
	{
		m_TrainVector_Fold2[i].erase(m_TrainVector_Fold2[i].begin() + 30, m_TrainVector_Fold2[i].end());
	}
	//cout << "m_TrainVector_Fold2: " << endl;
	//PrintVector2D(m_TrainVector_Fold2);
	for (int i = 0; i < m_TrainVector_Fold3.size(); ++i)
	{
		m_TrainVector_Fold3[i].erase(m_TrainVector_Fold3[i].begin() + 30, m_TrainVector_Fold3[i].end());
	}
	//cout << "m_TrainVector_Fold3: " << endl;
	//PrintVector2D(m_TrainVector_Fold3);

	//----------------------------------
	//VALIDATION
	for (int i = 0; i < m_ValidateVector_Fold1.size(); ++i)
	{
		m_ValidateVector_Fold1[i].erase(m_ValidateVector_Fold1[i].begin() + 30, m_ValidateVector_Fold1[i].end());
	}
	//cout << "m_ValidateVector_Fold1: " << endl;
	//PrintVector2D(m_ValidateVector_Fold1);
	
	for (int i = 0; i < m_ValidateVector_Fold2.size(); ++i)
	{
		m_ValidateVector_Fold2[i].erase(m_ValidateVector_Fold2[i].begin() + 30, m_ValidateVector_Fold2[i].end());
	}
	//cout << "m_ValidateVector_Fold2: " << endl;
	//PrintVector2D(m_ValidateVector_Fold2);

	for (int i = 0; i < m_ValidateVector_Fold3.size(); ++i)
	{
		m_ValidateVector_Fold3[i].erase(m_ValidateVector_Fold3[i].begin() + 30, m_ValidateVector_Fold3[i].end());
	}
	//cout << "m_ValidateVector_Fold3: " << endl;
	//PrintVector2D(m_ValidateVector_Fold3);
	
	//----------------------------------
	//TESTING
	for (int i = 0; i < m_TestVector_Fold1.size(); ++i)
	{
		m_TestVector_Fold1[i].erase(m_TestVector_Fold1[i].begin() + 30, m_TestVector_Fold1[i].end());
	}
	//cout << "m_TestVector_Fold1: " << endl;
	//PrintVector2D(m_TestVector_Fold1);
	
	for (int i = 0; i < m_TestVector_Fold2.size(); ++i)
	{
		m_TestVector_Fold2[i].erase(m_TestVector_Fold2[i].begin() + 30, m_TestVector_Fold2[i].end());
	}
	//cout << "m_TestVector_Fold2: " << endl;
	//PrintVector2D(m_TestVector_Fold2);

	for (int i = 0; i < m_TestVector_Fold3.size(); ++i)
	{
		m_TestVector_Fold3[i].erase(m_TestVector_Fold3[i].begin() + 30, m_TestVector_Fold3[i].end());
	}
	//cout << "m_TestVector_Fold3: " << endl;
	//PrintVector2D(m_TestVector_Fold3);
}

void DataStorage_16x20::ComputeAvgEigenFeatures()
{
	vector<double> sumDblVector;
	double returnArray[30];

	cout << "Fold1..." << endl;
	// cout << "Computing means..." << endl;
	ExtractAVG(m_TrainVector_Fold1, returnArray);
	for (int i = 0; i < 30; ++i)
	{
		// cout << i << ": " << returnArray[i] << endl;
		m_MeanEigenFeatures_Fold1.push_back(returnArray[i]);
	}
	DblVectorToE3Vector(m_MeanEigenFeatures_Fold1, 0, 1);

	// cout << "Computing variances..." << endl;
	ExtractVAR(m_TrainVector_Fold1, m_MeanEigenFeatures_Fold1, returnArray);
	for (int i = 0; i < 30; ++i)
	{
		// cout << i << ": " << returnArray[i] << endl;
		m_VarEigenFeatures_Fold1.push_back(returnArray[i]);
	}
	DblVectorToE3Vector(m_VarEigenFeatures_Fold1, 1, 1);

	cout << "Fold2..." << endl;
	// cout << "Computing means..." << endl;
	ExtractAVG(m_TrainVector_Fold2, returnArray);
	for (int i = 0; i < 30; ++i)
	{
		// cout << i << ": " << returnArray[i] << endl;
		m_MeanEigenFeatures_Fold2.push_back(returnArray[i]);
	}
	DblVectorToE3Vector(m_MeanEigenFeatures_Fold2, 0, 2);

	// cout << "Computing variances..." << endl;
	ExtractVAR(m_TrainVector_Fold2, m_MeanEigenFeatures_Fold2, returnArray);
	for (int i = 0; i < 30; ++i)
	{
		// cout << i << ": " << returnArray[i] << endl;
		m_VarEigenFeatures_Fold2.push_back(returnArray[i]);
	}
	DblVectorToE3Vector(m_VarEigenFeatures_Fold2, 1, 2);

	cout << "Fold3..." << endl;
	// cout << "Computing means..." << endl;
	ExtractAVG(m_TrainVector_Fold3, returnArray);
	for (int i = 0; i < 30; ++i)
	{
		// cout << i << ": " << returnArray[i] << endl;
		m_MeanEigenFeatures_Fold3.push_back(returnArray[i]);
	}
	DblVectorToE3Vector(m_MeanEigenFeatures_Fold3, 0, 3);
	
	// cout << "Computing variances..." << endl;
	ExtractVAR(m_TrainVector_Fold3, m_MeanEigenFeatures_Fold3, returnArray);
	for (int i = 0; i < 30; ++i)
	{
		// cout << i << ": " << returnArray[i] << endl;
		m_VarEigenFeatures_Fold3.push_back(returnArray[i]);
	}
	DblVectorToE3Vector(m_VarEigenFeatures_Fold3, 1, 3);
}

void DataStorage_48x60::ComputeAvgEigenFeatures()
{
	vector<double> sumDblVector;
	double returnArray[30];

	cout << "Fold1..." << endl;
	// cout << "Computing means..." << endl;
	ExtractAVG(m_TrainVector_Fold1, returnArray);
	for (int i = 0; i < 30; ++i)
	{
		// cout << i << ": " << returnArray[i] << endl;
		m_MeanEigenFeatures_Fold1.push_back(returnArray[i]);
	}
	DblVectorToE3Vector(m_MeanEigenFeatures_Fold1, 0, 1);

	// cout << "Computing variances..." << endl;
	ExtractVAR(m_TrainVector_Fold1, m_MeanEigenFeatures_Fold1, returnArray);
	for (int i = 0; i < 30; ++i)
	{
		// cout << i << ": " << returnArray[i] << endl;
		m_VarEigenFeatures_Fold1.push_back(returnArray[i]);
	}
	DblVectorToE3Vector(m_VarEigenFeatures_Fold1, 1, 1);

	cout << "Fold2..." << endl;
	// cout << "Computing means..." << endl;
	ExtractAVG(m_TrainVector_Fold2, returnArray);
	for (int i = 0; i < 30; ++i)
	{
		// cout << i << ": " << returnArray[i] << endl;
		m_MeanEigenFeatures_Fold2.push_back(returnArray[i]);
	}
	DblVectorToE3Vector(m_MeanEigenFeatures_Fold2, 0, 2);

	// cout << "Computing variances..." << endl;
	ExtractVAR(m_TrainVector_Fold2, m_MeanEigenFeatures_Fold2, returnArray);
	for (int i = 0; i < 30; ++i)
	{
		// cout << i << ": " << returnArray[i] << endl;
		m_VarEigenFeatures_Fold2.push_back(returnArray[i]);
	}
	DblVectorToE3Vector(m_VarEigenFeatures_Fold2, 1, 2);

	cout << "Fold3..." << endl;
	// cout << "Computing means..." << endl;
	ExtractAVG(m_TrainVector_Fold3, returnArray);
	for (int i = 0; i < 30; ++i)
	{
		// cout << i << ": " << returnArray[i] << endl;
		m_MeanEigenFeatures_Fold3.push_back(returnArray[i]);
	}
	DblVectorToE3Vector(m_MeanEigenFeatures_Fold3, 0, 3);
	
	// cout << "Computing variances..." << endl;
	ExtractVAR(m_TrainVector_Fold3, m_MeanEigenFeatures_Fold3, returnArray);
	for (int i = 0; i < 30; ++i)
	{
		// cout << i << ": " << returnArray[i] << endl;
		m_VarEigenFeatures_Fold3.push_back(returnArray[i]);
	}
	DblVectorToE3Vector(m_VarEigenFeatures_Fold3, 1, 3);
}

void DataStorage_48x60::DblVectorToE3Vector(vector<double> input, bool convertVariance, int fold)
{
	for (int i = 0; i < input.size(); ++i)
	{
		if(!convertVariance)
		{
			if(fold == 1)
			{
				m_MU_Fold1(i) = input[i];
				// cout << m_MU_Fold1(i) << " ";
			}
			else if(fold == 2)
			{
				m_MU_Fold2(i) = input[i];
				// cout << m_MU_Fold2(i) << " ";
			}
			else if(fold == 3)
			{
				m_MU_Fold3(i) = input[i];
				// cout << m_MU_Fold3(i) << " ";
			}
		}
		else
		{
			if(fold == 1)
			{
				m_VAR_Fold1(i) = input[i];
				// cout << m_VAR_Fold1(i) << " ";
			}
			else if(fold == 2)
			{
				m_VAR_Fold2(i) = input[i];
				// cout << m_VAR_Fold2(i) << " ";
			}
			else if(fold == 3)
			{
				m_VAR_Fold3(i) = input[i];
				// cout << m_VAR_Fold3(i) << " ";
			}
		}
	}
	// cout << endl;
}

void DataStorage_16x20::DblVectorToE3Vector(vector<double> input, bool convertVariance, int fold)
{
	for (int i = 0; i < input.size(); ++i)
	{
		if(!convertVariance)
		{
			if(fold == 1)
			{
				m_MU_Fold1(i) = input[i];
				// cout << m_MU_Fold1(i) << " ";
			}
			else if(fold == 2)
			{
				m_MU_Fold2(i) = input[i];
				// cout << m_MU_Fold2(i) << " ";
			}
			else if(fold == 3)
			{
				m_MU_Fold3(i) = input[i];
				// cout << m_MU_Fold3(i) << " ";
			}
		}
		else
		{
			if(fold == 1)
			{
				m_VAR_Fold1(i) = input[i];
				// cout << m_VAR_Fold1(i) << " ";
			}
			else if(fold == 2)
			{
				m_VAR_Fold2(i) = input[i];
				// cout << m_VAR_Fold2(i) << " ";
			}
			else if(fold == 3)
			{
				m_VAR_Fold3(i) = input[i];
				// cout << m_VAR_Fold3(i) << " ";
			}
		}
	}
	// cout << endl;
}

void ExtractAVG(vector<vector<double> > input, double (&array)[30])
{
	for (int i = 0; i < 30; ++i)
	{
		array[i] = 0;
	}

	for (int i = 0; i < input.size(); ++i)
	{
		for (int j = 0; j < input[i].size(); ++j)
		{
			array[j] += input[i][j];
		}
	}

	for (int i = 0; i < 30; ++i)
	{
		array[i] /= 30;
		//cout << "sumDbl: " << array[i] << endl;
	}
}
void ExtractVAR(vector<vector<double> > input, vector<double> means, double (&array)[30])
{
	for (int i = 0; i < 30; ++i)
	{
		array[i] = 0;
	}

	for (int i = 0; i < input.size(); ++i)
	{
		for (int j = 0; j < input[i].size(); ++j)
		{
			array[j] += (input[i][j]-means[j])*(input[i][j]-means[j]);
		}
	}

	for (int i = 0; i < 30; ++i)
	{
		array[i] /= 30;
		//cout << "sumDbl: " << array[i] << endl;
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