#include <bits/stdc++.h> 
#include <algorithm>
#include <climits>
#include <math.h>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <chrono>

using namespace std;

void printMatrix(int *matrix, int n) {
	for (int i = 0; i < n*n; i++) {
		if (i % n == 0) {
			cout << endl;
		}

		cout << matrix[i] << " ";
	}
}

bool equivChecker(int *ResultMatrixMatrix, int *expectedMatrix, int matrixSize) {
	for (int k = 0; k < matrixSize; k++) {
		if (expectedMatrix[k] != ResultMatrixMatrix[k]) {
			return false;
		}
	}
	return true;
}

// function to evaluate logarithm base-2
int calculateLog(int d) 
{ 
	int x = log2(d);
	// printf("Log %d is %d", d, x);
	return x; 
}

int * naive_serial(int *MatrixA, int *MatrixB, int *ResultMatrix, int n) {
	int numberOfEntries = n * n;
	for (int i = 0; i < numberOfEntries; i++) {
		ResultMatrix[i] = INT_MAX;
	}

	for (int row = 0; row < n; row++) {
		for (int collumn = 0; collumn < n; collumn++) {
			for (int k = 0; k < n; k++) {
				int index = row*n + collumn;
				ResultMatrix[index] = min(ResultMatrix[index], MatrixA[row*n + k] + MatrixB[k*n + collumn]);
			}
		}
	}
}

__global__ void kernel_1(int *MatrixA, int *MatrixB, int *ResultMatrix, int n) {
	int rowNumber = blockIdx.x/(n/blockDim.x);
	int firstIndexInRow = rowNumber*n;	
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	ResultMatrix[index] = 1;

	extern __shared__ int sharedData[];
	int numIndicesLoad = n/blockDim.x;
	if (n % blockDim.x > threadIdx.x) {
		numIndicesLoad++;
	}

	for (int i = 0; i < numIndicesLoad; i++) {
		sharedData[threadIdx.x + i] = MatrixA[firstIndexInRow + threadIdx.x + 1024*i];
	}

	__syncthreads();

	int resVal = INT_MAX;
	int collumn = index % n;

	//each thread computes the correct ResultMatrix for a given index in the 2D array
	for (int k = 0; k < n; k++) {
		int firstNum = sharedData[k];
		int secondNum = MatrixB[k*n + collumn];
		
		resVal = min(resVal, firstNum + secondNum);
	}

	ResultMatrix[index] = resVal;
}

std::pair<float,float> implementAlgorithm(int argc, char *argv[]) {
	int n;
	// Note: The width of the matrix is specified in the first line of the input test file
	ifstream myfile;
	myfile.open(argv[1]);
	myfile >> n;
	int matrix_size = n*n;
	int* ResultMatrix = (int*) malloc(matrix_size*sizeof(int));	
	int* MatrixA = (int*) malloc(matrix_size*sizeof(int));
	int* MatrixB = (int*) malloc(matrix_size*sizeof(int));
	int* serialResultMatrix = (int*) malloc(matrix_size*sizeof(int));
	for (int j = 0; j < matrix_size; j++) {
		myfile >> MatrixA[j];
		ResultMatrix[j] = INT_MAX;
		serialResultMatrix[j] = INT_MAX;
	}	
	for (int j = 0; j < matrix_size; j++) {
		myfile >> MatrixB[j];
	}
	int* expected = (int*) malloc(matrix_size*sizeof(int));
	for (int j = 0; j < matrix_size; j++) {
		myfile >> expected[j];
	}
	// int h = calculateLog(n);

	int* cudaMatrixA;
	int* cudaMatrixB;
	int* cudaResultMatrix;
	cudaMalloc((void **) &cudaMatrixA, sizeof(int) * matrix_size);
	cudaMalloc((void **) &cudaMatrixB, sizeof(int) * matrix_size);
	cudaMalloc((void **) &cudaResultMatrix, sizeof(int) * matrix_size);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	cudaMemcpy(cudaMatrixA, MatrixA, sizeof(int)*matrix_size, cudaMemcpyHostToDevice);
	cudaMemcpy(cudaMatrixB, MatrixB, sizeof(int)*matrix_size, cudaMemcpyHostToDevice);
	cudaMemcpy(cudaResultMatrix, ResultMatrix, sizeof(int)*matrix_size, cudaMemcpyHostToDevice);

	int thread_num = min(n, 1024);
	int thread_block_numb = matrix_size/thread_num;
	int size_shared_mem = n*sizeof(int);
	cudaEventRecord(start);
	kernel_1<<<thread_block_numb, thread_num, size_shared_mem>>>(cudaMatrixA, cudaMatrixB, cudaResultMatrix, n);

	
	cudaThreadSynchronize();	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cudaMemcpy(ResultMatrix, cudaResultMatrix, sizeof(int)*matrix_size, cudaMemcpyDeviceToHost);		


	auto begin = chrono::high_resolution_clock::now();
	kernel_1_serial(MatrixA, MatrixB, serialResultMatrix, n);	
	auto end = chrono::high_resolution_clock::now();
	auto dur = end - begin;
	auto serialTime = chrono::duration_cast<chrono::milliseconds>(dur).count();

	bool check = equivChecker(expected,ResultMatrix, matrix_size);

	cudaFree(cudaMatrixA);
	cudaFree(cudaMatrixB);
	cudaFree(cudaResultMatrix);

	if (check) {
		return std::make_pair(milliseconds,serialTime);
	} else {
		cout << "Result matrix does not equal the expected matrix given in the text file for " << argv[1] << endl;
	}
}


int main(int argc, char *argv[]) {
	int n;
	ifstream myfile;
	myfile.open(argv[1]);
	myfile >> n;
	int h = calculateLog(n);
	pair<float, float> p;
	for (int i = 0; i < h; i++) {
		p = implementAlgorithm(argc, argv);
	}
	cout << "Correct min-plus multiplication result for " << argv[1] << ", computed in " << p.first << " ms in parallel and " << p.second << " milliseconds in serial." << endl;

	return 0;
}
