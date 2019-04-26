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

int main()
{
	string inputString;
	do
	{	
			cout << endl
			     << "+==============================================================+\n"
				 << "|Select  0 to obtain training images (I_1...I_M)               |\n"
				 << "|Select  1 to compute average face vector (Psi)                |\n"
				 << "|Select  2 to compute matrix A ([Phi_i...Phi_M])               |\n"
				 << "|Select  3 to compute the eigenvectors/values of A^TA          |\n"
				 << "|Select  4 to project training eigenvalues (req: 0,1,2)        |\n"
				 << "|Select  5 to visualize the 10 largest & smallesteigenvectors  |\n"
			     << "|Select  6 to run facial recognition on fb_H                   |\n"
			     << "|Select  7 to run facial recognition on fb_H against fa2_H     |\n"
			     << "|Select -1 to exit                                             |\n"
			     << "+==============================================================+\n"
			     << endl
			     << "Choice: ";

			cin >> inputString;
	}while(inputString != "-1");
}

