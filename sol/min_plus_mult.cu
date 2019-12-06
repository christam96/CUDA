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

bool equivChecker(int *resultMatrix, int *expectedMatrix, int matrixSize) {
	for (int k = 0; k < matrixSize; k++) {
		if (expectedMatrix[k] != resultMatrix[k]) {
			return false;
		}
	}
	return true;
}

// function to evaluate logarithm base-2
int calculateLog(int d) 
{ 
	int result;
	int x = log2(d);
	//result = round(x);
	printf("Log %d is %d", d, x);
	cout<<endl;
	return x; 
} 

__global__ void min_plus_kernel_cache_first(int *matrix1, int *matrix2, int *result, int matrixWidth) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	extern __shared__ int sharedData[];
	sharedData[index] = matrix1[index];
	__syncthreads();

	int resultValue = INT_MAX;

	int col = index % matrixWidth;
	int row = index/matrixWidth;

	for (int k = 0; k < matrixWidth; k++) {
		int firstNum = sharedData[row*matrixWidth + k];
		int secondNum = matrix2[k*matrixWidth + col];
			
		resultValue = min(resultValue, firstNum + secondNum);
	}
	
	result[index] = resultValue;
}

__global__ void min_plus_kernel_cache_both(int *matrix1, int *matrix2, int *result, int matrixWidth) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = matrixWidth * matrixWidth;

	extern __shared__ int sharedData[];
	sharedData[index] = matrix1[index];
	sharedData[index + offset] = matrix2[index];
	__syncthreads();

	int resultValue = INT_MAX;

	int col = index % matrixWidth;
	int row = index/matrixWidth;
	
	//each thread computes the correct result for a given index in the 2D array
	for (int k = 0; k < matrixWidth; k++) {
		int firstNum = sharedData[row*matrixWidth + k];
		int secondNum = sharedData[k*matrixWidth + col + offset];
		//int firstNum = sharedData[k];
		//int secondNum = matrix2[k*matrixWidth + col];
			
		resultValue = min(resultValue, firstNum + secondNum);
	}
	
	result[index] = resultValue;
}

__global__ void min_plus(int *matrix1, int *matrix2, int *result, int matrixWidth) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int rowNumber = blockIdx.x/(matrixWidth/blockDim.x);
	int firstIndexInRow = rowNumber*matrixWidth;
	
	result[index] = 1;

	extern __shared__ int sharedData[];
	int numberOfIndicesToLoad = matrixWidth/blockDim.x;
	if (matrixWidth % blockDim.x > threadIdx.x) {
		numberOfIndicesToLoad++;
	}

	for (int i = 0; i < numberOfIndicesToLoad; i++) {
		sharedData[threadIdx.x + i*1024] = matrix1[firstIndexInRow + threadIdx.x + 1024*i];
	}

	__syncthreads();

	int resultValue = INT_MAX;

	int col = index % matrixWidth;

	//each thread computes the correct result for a given index in the 2D array
	for (int k = 0; k < matrixWidth; k++) {
		int firstNum = sharedData[k];
		int secondNum = matrix2[k*matrixWidth + col];
		
		resultValue = min(resultValue, firstNum + secondNum);
	}

	result[index] = resultValue;
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

void implementAlgorithm(int argc, char *argv[]) {
	for (int i = 1; i < argc; i++) {
		int matrixWidth;

		//first number in file is matrix width
		ifstream myfile;
		myfile.open(argv[i]);
		myfile >> matrixWidth;

		//load first input matrix
		int sizeOfMatrix = matrixWidth*matrixWidth;
		int* matrix1 = (int*) malloc(sizeOfMatrix*sizeof(int));
		int* matrix2 = (int*) malloc(sizeOfMatrix*sizeof(int));
		int* result = (int*) malloc(sizeOfMatrix*sizeof(int));
		int* serialResult = (int*) malloc(sizeOfMatrix*sizeof(int));
		for (int j = 0; j < sizeOfMatrix; j++) {
			myfile >> matrix1[j];
			result[j] = INT_MAX;
			serialResult[j] = INT_MAX;
		}	

		for (int j = 0; j < sizeOfMatrix; j++) {
			myfile >> matrix2[j];
		}

		cout << "Matrix 1" << endl;
		printMatrix(matrix1, matrixWidth);
		
		cout << endl << "Matrix 2" << endl;
		printMatrix(matrix2, matrixWidth);
		

		//load expected result
		int* expected = (int*) malloc(sizeOfMatrix*sizeof(int));
		for (int j = 0; j < sizeOfMatrix; j++) {
			myfile >> expected[j];
		}

		int h = calculateLog(matrixWidth);

		cout << endl << "Expected" << endl;
		printMatrix(expected, matrixWidth);

		int* cudaMatrix1;
		int* cudaMatrix2;
		int* cudaResult;
		cudaMalloc((void **) &cudaMatrix1, sizeof(int) * sizeOfMatrix);
		cudaMalloc((void **) &cudaMatrix2, sizeof(int) * sizeOfMatrix);
		cudaMalloc((void **) &cudaResult, sizeof(int) * sizeOfMatrix);

		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		
		cudaMemcpy(cudaMatrix1, matrix1, sizeof(int)*sizeOfMatrix, cudaMemcpyHostToDevice);
		cudaMemcpy(cudaMatrix2, matrix2, sizeof(int)*sizeOfMatrix, cudaMemcpyHostToDevice);
		cudaMemcpy(cudaResult, result, sizeof(int)*sizeOfMatrix, cudaMemcpyHostToDevice);

		//cout << endl << "Result" << endl;
		//printMatrix(result, matrixWidth);

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
		// 				min_plus_kernel_cache_both<<<numberOfThreadBlocks, numberOfThreads, sharedMemorySize>>>(cudaMatrix1, cudaMatrix2, cudaResult, matrixWidth);
		// 			} else {
		// 				int sharedMemorySize = sizeOfMatrix*sizeof(int);
		// 				cudaEventRecord(start);
		// 				cout << "CACHING FIRST MATRIX:" << endl;
		// 				min_plus_kernel_cache_first<<<numberOfThreadBlocks, numberOfThreads, sharedMemorySize>>>(cudaMatrix1, cudaMatrix2, cudaResult, matrixWidth);
		// 			}
		
		// 		} else {
		// 			cout << "CACHING ROW IN MATRIX:" << endl;
		// 			int numberOfThreads = min(matrixWidth, 1024);
		// 			int numberOfThreadBlocks = sizeOfMatrix/numberOfThreads;
		// 			int sharedMemorySize = matrixWidth*sizeof(int);
		// 			cudaEventRecord(start);
		// 			min_plus<<<numberOfThreadBlocks, numberOfThreads, sharedMemorySize>>>(cudaMatrix1, cudaMatrix2, cudaResult, matrixWidth);
		// 		}
		// 		for (int j = 0; j < sizeOfMatrix; j++) {
		// 			cudaMatrix2[j] = cudaResult[j];
		// 			cudaResult[j] = INT_MAX;
		// 		}
		// 	} else {
		// 		if (matrixWidth < 128) {
		// 			int numberOfThreadBlocks = ceil(sizeOfMatrix/1024.0);
		// 			int numberOfThreads = min(1024, sizeOfMatrix);
		
		// 			if (matrixWidth < 64) {
		// 				int sharedMemorySize = sizeOfMatrix*2*sizeof(int);
		// 				cudaEventRecord(start);
		// 				cout << "CACHING BOTH MATRICES:" << endl;
		// 				min_plus_kernel_cache_both<<<numberOfThreadBlocks, numberOfThreads, sharedMemorySize>>>(cudaMatrix1, cudaMatrix2, cudaResult, matrixWidth);
		// 			} else {
		// 				int sharedMemorySize = sizeOfMatrix*sizeof(int);
		// 				cudaEventRecord(start);
		// 				cout << "CACHING FIRST MATRIX:" << endl;
		// 				min_plus_kernel_cache_first<<<numberOfThreadBlocks, numberOfThreads, sharedMemorySize>>>(cudaMatrix1, cudaMatrix2, cudaResult, matrixWidth);
		// 			}
		
		// 		} else {
		// 			cout << "CACHING ROW IN MATRIX:" << endl;
		// 			int numberOfThreads = min(matrixWidth, 1024);
		// 			int numberOfThreadBlocks = sizeOfMatrix/numberOfThreads;
		// 			int sharedMemorySize = matrixWidth*sizeof(int);
		// 			cudaEventRecord(start);
		// 			min_plus<<<numberOfThreadBlocks, numberOfThreads, sharedMemorySize>>>(cudaMatrix1, cudaMatrix2, cudaResult, matrixWidth);
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
				min_plus_kernel_cache_both<<<numberOfThreadBlocks, numberOfThreads, sharedMemorySize>>>(cudaMatrix1, cudaMatrix2, cudaResult, matrixWidth);
			} else {
				int sharedMemorySize = sizeOfMatrix*sizeof(int);
				cudaEventRecord(start);
				cout << "CACHING FIRST MATRIX:" << endl;
				min_plus_kernel_cache_first<<<numberOfThreadBlocks, numberOfThreads, sharedMemorySize>>>(cudaMatrix1, cudaMatrix2, cudaResult, matrixWidth);
			}

		} else {
			cout << "CACHING ROW IN MATRIX:" << endl;
			int numberOfThreads = min(matrixWidth, 1024);
			int numberOfThreadBlocks = sizeOfMatrix/numberOfThreads;
			int sharedMemorySize = matrixWidth*sizeof(int);
			cudaEventRecord(start);
			min_plus<<<numberOfThreadBlocks, numberOfThreads, sharedMemorySize>>>(cudaMatrix1, cudaMatrix2, cudaResult, matrixWidth);
		}
		
		cudaThreadSynchronize();	
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventSynchronize(stop);

		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		cudaMemcpy(result, cudaResult, sizeof(int)*sizeOfMatrix, cudaMemcpyDeviceToHost);		


		auto begin = chrono::high_resolution_clock::now();
		min_plus_serial(matrix1, matrix2, serialResult, matrixWidth);	
		auto end = chrono::high_resolution_clock::now();
		auto dur = end - begin;
		auto serialTime = chrono::duration_cast<chrono::milliseconds>(dur).count();

		// //validate result
		// bool ifEquiv = true;
		// for (int k = 0; k < sizeOfMatrix; k++) {
		// 	if (expected[k] != result[k]) {
		// 		ifEquiv = false;
		// 		break;
		// 	}
		// }
		bool check = equivChecker(expected,result, sizeOfMatrix);

		cudaFree(cudaMatrix1);
		cudaFree(cudaMatrix2);
		cudaFree(cudaResult);

		if (check) {
			cout << "Computed min-plus multiplication for " << argv[i] << " correctly in " << milliseconds << " ms in parallel and " << serialTime << " milliseconds in serial." << endl;
		} else {
			for (int k = 0; k < sizeOfMatrix; k++) {
				if (k % matrixWidth == 0) {
					cout << endl;
				}

				cout << result[k] << " ";
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
