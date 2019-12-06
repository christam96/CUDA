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

void printMatrix(int *matrix, int matrixWidth) {
	for (int i = 0; i < matrixWidth*matrixWidth; i++) {
		if (i % matrixWidth == 0) {
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
	printf("Log %d is %d", d, x);
	return x; 
} 

__global__ void min_plus_kernel_cache_first(int *MatrixA, int *MatrixB, int *ResultMatrix, int matrixWidth) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	extern __shared__ int sharedData[];
	sharedData[index] = MatrixA[index];
	__syncthreads();

	int resVal = INT_MAX;

	int col = index % matrixWidth;
	int row = index/matrixWidth;

	for (int k = 0; k < matrixWidth; k++) {
		int firstNum = sharedData[row*matrixWidth + k];
		int secondNum = MatrixB[k*matrixWidth + col];
			
		resVal = min(resVal, firstNum + secondNum);
	}
	
	ResultMatrix[index] = resVal;
}

__global__ void min_plus_kernel_cache_both(int *MatrixA, int *MatrixB, int *ResultMatrix, int matrixWidth) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = matrixWidth * matrixWidth;

	extern __shared__ int sharedData[];
	sharedData[index] = MatrixA[index];
	sharedData[index + offset] = MatrixB[index];
	__syncthreads();

	int resVal = INT_MAX;

	int col = index % matrixWidth;
	int row = index/matrixWidth;
	
	//each thread computes the correct ResultMatrix for a given index in the 2D array
	for (int k = 0; k < matrixWidth; k++) {
		int firstNum = sharedData[row*matrixWidth + k];
		int secondNum = sharedData[k*matrixWidth + col + offset];
		//int firstNum = sharedData[k];
		//int secondNum = MatrixB[k*matrixWidth + col];
			
		resVal = min(resVal, firstNum + secondNum);
	}
	
	ResultMatrix[index] = resVal;
}

__global__ void min_plus(int *MatrixA, int *MatrixB, int *ResultMatrix, int matrixWidth) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int rowNumber = blockIdx.x/(matrixWidth/blockDim.x);
	int firstIndexInRow = rowNumber*matrixWidth;
	
	ResultMatrix[index] = 1;

	extern __shared__ int sharedData[];
	int numberOfIndicesToLoad = matrixWidth/blockDim.x;
	if (matrixWidth % blockDim.x > threadIdx.x) {
		numberOfIndicesToLoad++;
	}

	for (int i = 0; i < numberOfIndicesToLoad; i++) {
		sharedData[threadIdx.x + i*1024] = MatrixA[firstIndexInRow + threadIdx.x + 1024*i];
	}

	__syncthreads();

	int resVal = INT_MAX;

	int col = index % matrixWidth;

	//each thread computes the correct ResultMatrix for a given index in the 2D array
	for (int k = 0; k < matrixWidth; k++) {
		int firstNum = sharedData[k];
		int secondNum = MatrixB[k*matrixWidth + col];
		
		resVal = min(resVal, firstNum + secondNum);
	}

	ResultMatrix[index] = resVal;
}

void min_plus_serial(int *MatrixA, int *MatrixB, int *ResultMatrix, int matrixWidth) {
	int numberOfEntries = matrixWidth * matrixWidth;
	for (int i = 0; i < numberOfEntries; i++) {
		ResultMatrix[i] = INT_MAX;
	}

	for (int row = 0; row < matrixWidth; row++) {
		for (int col = 0; col < matrixWidth; col++) {
			for (int k = 0; k < matrixWidth; k++) {
				int index = row*matrixWidth + col;
				ResultMatrix[index] = min(ResultMatrix[index], MatrixA[row*matrixWidth + k] + MatrixB[k*matrixWidth + col]);
			}
		}
	}
}

void implementAlgorithm(int argc, char *argv[]) {
	for (int i = 1; i < argc; i++) {
		int matrixWidth;

		//first number in file is matrix width
		ifstream myfile;
		myfile.open(argv[i]);
		myfile >> matrixWidth;

		//load first input matrix
		int sizeOfMatrix = matrixWidth*matrixWidth;
		int* MatrixA = (int*) malloc(sizeOfMatrix*sizeof(int));
		int* MatrixB = (int*) malloc(sizeOfMatrix*sizeof(int));
		int* ResultMatrix = (int*) malloc(sizeOfMatrix*sizeof(int));
		int* serialResultMatrix = (int*) malloc(sizeOfMatrix*sizeof(int));
		for (int j = 0; j < sizeOfMatrix; j++) {
			myfile >> MatrixA[j];
			ResultMatrix[j] = INT_MAX;
			serialResultMatrix[j] = INT_MAX;
		}	

		for (int j = 0; j < sizeOfMatrix; j++) {
			myfile >> MatrixB[j];
		}

		cout << "Matrix 1" << endl;
		printMatrix(MatrixA, matrixWidth);
		
		cout << endl << "Matrix 2" << endl;
		printMatrix(MatrixB, matrixWidth);
		

		//load expected ResultMatrix
		int* expected = (int*) malloc(sizeOfMatrix*sizeof(int));
		for (int j = 0; j < sizeOfMatrix; j++) {
			myfile >> expected[j];
		}

		int h = calculateLog(matrixWidth);

		cout << endl << "Expected" << endl;
		printMatrix(expected, matrixWidth);

		int* cudaMatrixA;
		int* cudaMatrixB;
		int* cudaResultMatrix;
		cudaMalloc((void **) &cudaMatrixA, sizeof(int) * sizeOfMatrix);
		cudaMalloc((void **) &cudaMatrixB, sizeof(int) * sizeOfMatrix);
		cudaMalloc((void **) &cudaResultMatrix, sizeof(int) * sizeOfMatrix);

		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		
		cudaMemcpy(cudaMatrixA, MatrixA, sizeof(int)*sizeOfMatrix, cudaMemcpyHostToDevice);
		cudaMemcpy(cudaMatrixB, MatrixB, sizeof(int)*sizeOfMatrix, cudaMemcpyHostToDevice);
		cudaMemcpy(cudaResultMatrix, ResultMatrix, sizeof(int)*sizeOfMatrix, cudaMemcpyHostToDevice);

		//cout << endl << "ResultMatrix" << endl;
		//printMatrix(ResultMatrix, matrixWidth);

		// ALTERNATIVE KERNEL CALL
		// for (int i = 0; i < h; i++) {
		// 	if (i < h-1) {
		// 		if (matrixWidth < 128) {
		// 			int numberOfThreadBlocks = ceil(sizeOfMatrix/1024.0);
		// 			int numberOfThreads = min(1024, sizeOfMatrix);
		
		// 			if (matrixWidth < 64) {
		// 				int sharedMemorySize = sizeOfMatrix*2*sizeof(int);
		// 				cudaEventRecord(start);
		// 				cout << "CACHING BOTH MATRICES:" << endl;
		// 				min_plus_kernel_cache_both<<<numberOfThreadBlocks, numberOfThreads, sharedMemorySize>>>(cudaMatrixA, cudaMatrixB, cudaResultMatrix, matrixWidth);
		// 			} else {
		// 				int sharedMemorySize = sizeOfMatrix*sizeof(int);
		// 				cudaEventRecord(start);
		// 				cout << "CACHING FIRST MATRIX:" << endl;
		// 				min_plus_kernel_cache_first<<<numberOfThreadBlocks, numberOfThreads, sharedMemorySize>>>(cudaMatrixA, cudaMatrixB, cudaResultMatrix, matrixWidth);
		// 			}
		
		// 		} else {
		// 			cout << "CACHING ROW IN MATRIX:" << endl;
		// 			int numberOfThreads = min(matrixWidth, 1024);
		// 			int numberOfThreadBlocks = sizeOfMatrix/numberOfThreads;
		// 			int sharedMemorySize = matrixWidth*sizeof(int);
		// 			cudaEventRecord(start);
		// 			min_plus<<<numberOfThreadBlocks, numberOfThreads, sharedMemorySize>>>(cudaMatrixA, cudaMatrixB, cudaResultMatrix, matrixWidth);
		// 		}
		// 		for (int j = 0; j < sizeOfMatrix; j++) {
		// 			cudaMatrixB[j] = cudaResultMatrix[j];
		// 			cudaResultMatrix[j] = INT_MAX;
		// 		}
		// 	} else {
		// 		if (matrixWidth < 128) {
		// 			int numberOfThreadBlocks = ceil(sizeOfMatrix/1024.0);
		// 			int numberOfThreads = min(1024, sizeOfMatrix);
		
		// 			if (matrixWidth < 64) {
		// 				int sharedMemorySize = sizeOfMatrix*2*sizeof(int);
		// 				cudaEventRecord(start);
		// 				cout << "CACHING BOTH MATRICES:" << endl;
		// 				min_plus_kernel_cache_both<<<numberOfThreadBlocks, numberOfThreads, sharedMemorySize>>>(cudaMatrixA, cudaMatrixB, cudaResultMatrix, matrixWidth);
		// 			} else {
		// 				int sharedMemorySize = sizeOfMatrix*sizeof(int);
		// 				cudaEventRecord(start);
		// 				cout << "CACHING FIRST MATRIX:" << endl;
		// 				min_plus_kernel_cache_first<<<numberOfThreadBlocks, numberOfThreads, sharedMemorySize>>>(cudaMatrixA, cudaMatrixB, cudaResultMatrix, matrixWidth);
		// 			}
		
		// 		} else {
		// 			cout << "CACHING ROW IN MATRIX:" << endl;
		// 			int numberOfThreads = min(matrixWidth, 1024);
		// 			int numberOfThreadBlocks = sizeOfMatrix/numberOfThreads;
		// 			int sharedMemorySize = matrixWidth*sizeof(int);
		// 			cudaEventRecord(start);
		// 			min_plus<<<numberOfThreadBlocks, numberOfThreads, sharedMemorySize>>>(cudaMatrixA, cudaMatrixB, cudaResultMatrix, matrixWidth);
		// 		}
		// 	}
		// }

		// INITIAL KERNEL
		if (matrixWidth < 128) {
			int numberOfThreadBlocks = ceil(sizeOfMatrix/1024.0);
			int numberOfThreads = min(1024, sizeOfMatrix);

			if (matrixWidth < 64) {
				int sharedMemorySize = sizeOfMatrix*2*sizeof(int);
				cudaEventRecord(start);
				cout << "CACHING BOTH MATRICES:" << endl;
				min_plus_kernel_cache_both<<<numberOfThreadBlocks, numberOfThreads, sharedMemorySize>>>(cudaMatrixA, cudaMatrixB, cudaResultMatrix, matrixWidth);
			} else {
				int sharedMemorySize = sizeOfMatrix*sizeof(int);
				cudaEventRecord(start);
				cout << "CACHING FIRST MATRIX:" << endl;
				min_plus_kernel_cache_first<<<numberOfThreadBlocks, numberOfThreads, sharedMemorySize>>>(cudaMatrixA, cudaMatrixB, cudaResultMatrix, matrixWidth);
			}

		} else {
			cout << "CACHING ROW IN MATRIX:" << endl;
			int numberOfThreads = min(matrixWidth, 1024);
			int numberOfThreadBlocks = sizeOfMatrix/numberOfThreads;
			int sharedMemorySize = matrixWidth*sizeof(int);
			cudaEventRecord(start);
			min_plus<<<numberOfThreadBlocks, numberOfThreads, sharedMemorySize>>>(cudaMatrixA, cudaMatrixB, cudaResultMatrix, matrixWidth);
		}
		
		cudaThreadSynchronize();	
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventSynchronize(stop);

		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		cudaMemcpy(ResultMatrix, cudaResultMatrix, sizeof(int)*sizeOfMatrix, cudaMemcpyDeviceToHost);		


		auto begin = chrono::high_resolution_clock::now();
		min_plus_serial(MatrixA, MatrixB, serialResultMatrix, matrixWidth);	
		auto end = chrono::high_resolution_clock::now();
		auto dur = end - begin;
		auto serialTime = chrono::duration_cast<chrono::milliseconds>(dur).count();

		// //validate ResultMatrix
		// bool ifEquiv = true;
		// for (int k = 0; k < sizeOfMatrix; k++) {
		// 	if (expected[k] != ResultMatrix[k]) {
		// 		ifEquiv = false;
		// 		break;
		// 	}
		// }
		bool check = equivChecker(expected,ResultMatrix, sizeOfMatrix);

		cudaFree(cudaMatrixA);
		cudaFree(cudaMatrixB);
		cudaFree(cudaResultMatrix);

		if (check) {
			cout << "Computed min-plus multiplication for " << argv[i] << " correctly in " << milliseconds << " ms in parallel and " << serialTime << " milliseconds in serial." << endl;
		} else {
			for (int k = 0; k < sizeOfMatrix; k++) {
				if (k % matrixWidth == 0) {
					cout << endl;
				}

				cout << ResultMatrix[k] << " ";
			}

			cout << "Error computing min-plus for " << argv[i] << endl;
			//cout << endl << cudaGetErrorString(cudaGetLastError()) << endl;
			//cudaError_t error = cudaGetLastError();
		//	cout << cudaGetLastError() << endl;
		}
	}
}


int main(int argc, char *argv[]) {
	implementAlgorithm(argc, argv);
	return 0;
}
