#include <cstdio>
#include <iostream>
#include <fstream>
#include <climits>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <bits/stdc++.h> 
#include <math.h>

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

__global__ void min_plus_kernel_cache_first(int *MatrixA, int *MatrixB, int *ResultMatrix, int n) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	extern __shared__ int sharedData[];
	sharedData[index] = MatrixA[index];
	__syncthreads();

	int resVal = INT_MAX;

	int col = index % n;
	int row = index/n;

	for (int k = 0; k < n; k++) {
		int firstNum = sharedData[row*n + k];
		int secondNum = MatrixB[k*n + col];
			
		resVal = min(resVal, firstNum + secondNum);
	}
	
	ResultMatrix[index] = resVal;
}

__global__ void min_plus_kernel_cache_both(int *MatrixA, int *MatrixB, int *ResultMatrix, int n) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = n * n;

	extern __shared__ int sharedData[];
	sharedData[index] = MatrixA[index];
	sharedData[index + offset] = MatrixB[index];
	__syncthreads();

	int resVal = INT_MAX;

	int col = index % n;
	int row = index/n;
	
	//each thread computes the correct ResultMatrix for a given index in the 2D array
	for (int k = 0; k < n; k++) {
		int firstNum = sharedData[row*n + k];
		int secondNum = sharedData[k*n + col + offset];
		//int firstNum = sharedData[k];
		//int secondNum = MatrixB[k*n + col];
			
		resVal = min(resVal, firstNum + secondNum);
	}
	
	ResultMatrix[index] = resVal;
}

__global__ void min_plus(int *MatrixA, int *MatrixB, int *ResultMatrix, int n) {
	int rowNumber = blockIdx.x/(n/blockDim.x);
	int firstIndexInRow = rowNumber*n;
	
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	
	ResultMatrix[index] = 1;

	extern __shared__ int sharedData[];
	int numberOfIndicesToLoad = n/blockDim.x;
	if (n % blockDim.x > threadIdx.x) {
		numberOfIndicesToLoad++;
	}

	for (int i = 0; i < numberOfIndicesToLoad; i++) {
		sharedData[threadIdx.x + i*1024] = MatrixA[firstIndexInRow + threadIdx.x + 1024*i];
	}

	__syncthreads();

	int resVal = INT_MAX;

	int col = index % n;

	//each thread computes the correct ResultMatrix for a given index in the 2D array
	for (int k = 0; k < n; k++) {
		int firstNum = sharedData[k];
		int secondNum = MatrixB[k*n + col];
		
		resVal = min(resVal, firstNum + secondNum);
	}

	ResultMatrix[index] = resVal;
}

int * min_plus_serial(int *MatrixA, int *MatrixB, int *ResultMatrix, int n) {
	int numberOfEntries = n * n;
	for (int i = 0; i < numberOfEntries; i++) {
		ResultMatrix[i] = INT_MAX;
	}

	for (int row = 0; row < n; row++) {
		for (int col = 0; col < n; col++) {
			for (int k = 0; k < n; k++) {
				int index = row*n + col;
				ResultMatrix[index] = min(ResultMatrix[index], MatrixA[row*n + k] + MatrixB[k*n + col]);
			}
		}
	}
}

std::pair<int,int> implementAlgorithm(int argc, char *argv[]) {
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

	// cout << "Matrix 1" << endl;
	// printMatrix(MatrixA, n);
	
	// cout << endl << "Matrix 2" << endl;
	// printMatrix(MatrixB, n);
	

	//load expected ResultMatrix
	int* expected = (int*) malloc(matrix_size*sizeof(int));
	for (int j = 0; j < matrix_size; j++) {
		myfile >> expected[j];
	}

	int h = calculateLog(n);

	// cout << endl << "Expected" << endl;
	// printMatrix(expected, n);

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

	//cout << endl << "ResultMatrix" << endl;
	//printMatrix(ResultMatrix, n);

	if (n < 32) {
		int thread_block_numb = ceil(matrix_size/1024.0);
		int thread_num = min(1024, matrix_size);

		if (n < 64) {
			int size_shared_mem = matrix_size*2*sizeof(int);
			cudaEventRecord(start);
			min_plus_kernel_cache_both<<<thread_block_numb, thread_num, size_shared_mem>>>(cudaMatrixA, cudaMatrixB, cudaResultMatrix, n);
		} else {
			int size_shared_mem = matrix_size*sizeof(int);
			cudaEventRecord(start);
			min_plus_kernel_cache_first<<<thread_block_numb, thread_num, size_shared_mem>>>(cudaMatrixA, cudaMatrixB, cudaResultMatrix, n);
		}
	} else {
		int thread_num = min(n, 1024);
		int thread_block_numb = matrix_size/thread_num;
		int size_shared_mem = n*sizeof(int);
		cudaEventRecord(start);
		min_plus<<<thread_block_numb, thread_num, size_shared_mem>>>(cudaMatrixA, cudaMatrixB, cudaResultMatrix, n);
	}
	
	cudaThreadSynchronize();	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cudaMemcpy(ResultMatrix, cudaResultMatrix, sizeof(int)*matrix_size, cudaMemcpyDeviceToHost);		


	auto begin = chrono::high_resolution_clock::now();
	min_plus_serial(MatrixA, MatrixB, serialResultMatrix, n);	
	auto end = chrono::high_resolution_clock::now();
	auto dur = end - begin;
	auto serialTime = chrono::duration_cast<chrono::milliseconds>(dur).count();

	// //validate ResultMatrix
	// bool ifEquiv = true;
	// for (int k = 0; k < matrix_size; k++) {
	// 	if (expected[k] != ResultMatrix[k]) {
	// 		ifEquiv = false;
	// 		break;
	// 	}
	// }
	bool check = equivChecker(expected,ResultMatrix, matrix_size);

	cudaFree(cudaMatrixA);
	cudaFree(cudaMatrixB);
	cudaFree(cudaResultMatrix);

	if (check) {
		// cout << "Computed min-plus multiplication for " << argv[1] << " correctly in " << milliseconds << " ms in parallel and " << serialTime << " milliseconds in serial." << endl;
		return std::make_pair(milliseconds,serialTime);
	} else {
		for (int k = 0; k < matrix_size; k++) {
			if (k % n == 0) {
				cout << endl;
			}

			cout << ResultMatrix[k] << " ";
		}
		cout << "Error computing min-plus for " << argv[1] << endl;
		//cout << endl << cudaGetErrorString(cudaGetLastError()) << endl;
		//cudaError_t error = cudaGetLastError();
	//	cout << cudaGetLastError() << endl;
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
	cout << "Computed min-plus multiplication for " << argv[1] << " correctly in " << p.first << " ms in parallel and " << p.second << " milliseconds in serial." << endl;

	return 0;
}
