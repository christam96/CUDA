#include <iostream>
#include<bits/stdc++.h>
#include <string>
#include <sstream> // for std::stringstream
#include <cstdlib> // for exit()
#include <time.h>       /* time */
#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
using namespace std;
void print(int *matrix, int matrixWidth) {
        for (int i = 0; i < matrixWidth*matrixWidth; i++) {
                if (i % matrixWidth == 0) {
                        cout << endl;
                }

                cout << matrix[i] << " ";
        }
}

void min_plus_serial(int *matrix1, int *matrix2, int *result, int matrixWidth) {
        int numberOfEntries = matrixWidth * matrixWidth;
        for (int i = 0; i < numberOfEntries; i++) {
                result[i] = INT_MAX;
        }

        for (int row = 0; row < matrixWidth; row++) {
                for (int col = 0; col < matrixWidth; col++) {
                        for (int k = 0; k < matrixWidth; k++) {
                                int index = row*matrixWidth + col;
                                result[index] = min(result[index], matrix1[row*matrixWidth + k] + matrix2[k*matrixWidth + col]);
                        }
                }
        }
}

// function to evaluate logarithm base-2
int calculateLog(int d) 
{ 
	int result;
	int x = log2(d);
	//result = round(x);
	// printf("Log %d is %d", d, x);
	// cout<<endl;
	return x; 
} 


int main(int argc, char *argv[])
{
	if (argc <= 1)
	{
		// On some operating systems, argv[0] can end up as an empty string instead of the program's name.
		// We'll conditionalize our response on whether argv[0] is empty or not.
		if (argv[0])
			std::cout << "Usage: " << argv[0] << " <number>" << '\n';
		else
			std::cout << "Usage: <program name> <number>" << '\n';
 
		exit(1);
	}
 
	std::stringstream convert(argv[1]); // set up a stringstream variable named convert, initialized with the input from argv[1]
 
	int matrixSize;
	if (!(convert >> matrixSize)) // do the conversion
		matrixSize = 0; // if conversion fails, set myint to a default value
 
	std::cout << "Got integer: " << matrixSize << '\n';
 	
	/* initialize random seed: */
  	srand (time(NULL));
	
	int sizeOfMatrix = matrixSize*matrixSize;
        int* matrix1 = (int*) malloc(sizeOfMatrix*sizeof(int));
	
	// cout<<"check"<<endl;

	for (int j = 0; j < sizeOfMatrix; j++) {
                        /* generate secret number between 1 and 10: */
		        int iSecret = rand() % 10;
                        matrix1[j]  = iSecret;
			
       }

	for (int i = 0; i < matrixSize; i++) {
		for (int j = 0; j < matrixSize; j++) {
			if (i==j) matrix1[i*matrixSize+j] = 0;

		}
	}

	int h = calculateLog(matrixSize);

	// cout<<"Input Matrix";	
	// print(matrix1, matrixSize);
	

	int* matrix2 = (int*) malloc(sizeOfMatrix*sizeof(int));
	int* matrix3 = (int*) malloc(sizeOfMatrix*sizeof(int));

	for (int i = 0; i < sizeOfMatrix; i++) {
		matrix2[i] = matrix1[i];
	}

	// cout<<endl<<"Matrix2:";
	// print(matrix2, matrixSize);

	// for (int i = 0; i < h; i++) {
	// 	if (i < h-1) {
	// 		min_plus_serial(matrix1,matrix2,matrix3,matrixSize);
	// 		for (int j = 0; j < matrixSize ; j++) {
	// 			matrix2[j] = matrix3[j];
	// 			matrix3[j] = INT_MAX;
	// 		}
	// 	} else {
	// 		min_plus_serial(matrix1,matrix2,matrix3,matrixSize);
	// 	}
	// }

	// cout<<endl<<"Final Result after logn iterations: "<<endl;
    //     print(matrix3,matrixSize);

	
	min_plus_serial(matrix1,matrix2,matrix3,matrixSize);

	// cout<<endl<<"Final Result: "<<endl;
	// print(matrix3,matrixSize);

	ofstream myfile ("example.txt");
	if (myfile.is_open()) {
   		myfile << matrixSize;
		myfile << "\n";
		for (int i=0;i<matrixSize*matrixSize;i++) {
			myfile << matrix1[i];
			myfile << " ";	
		}
		myfile << "\n";
		for (int i=0;i<matrixSize*matrixSize;i++) {
                        myfile << matrix1[i];
			myfile << " ";
        }
        myfile << "\n";
		for (int i=0;i<matrixSize*matrixSize;i++) {
            myfile << matrix3[i];
			myfile << " ";
        }
        myfile << "\n";
	    myfile.close();
 	}
  	else cout << "Unable to open file";	

	return 0;
}
