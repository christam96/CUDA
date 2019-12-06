#include <cstdio>
#include <iostream>
#include <fstream>
#include <climits>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <bits/stdc++.h> 

using namespace std;

void printMatrix(int *matrix, int matrixWidth) {
	for (int i = 0; i < matrixWidth*matrixWidth; i++) {
		if (i % matrixWidth == 0) {
			cout << endl;
		}

		cout << matrix[i] << " ";
	}
}

// function to evaluate logarithm base-10 
int calculateLog10(double d) 
{ 
	int result;
	double x = log10(d);
	result = round(x);
	printf("Log %f is %d", d, result);
	return result; 
} 

__global__ void min_plus_kernel_cache_first(int *matrix1, int *matrix2, int *result, int matrixWidth) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	extern __shared__ int sData[];
	sData[index] = matrix1[index];
	__syncthreads();

	int resultValue = INT_MAX;

	int col = index % matrixWidth;
	int row = index/matrixWidth;

	for (int k = 0; k < matrixWidth; k++) {
		int firstNum = sData[row*matrixWidth + k];
		int secondNum = matrix2[k*matrixWidth + col];
			
		resultValue = min(resultValue, firstNum + secondNum);
	}
	
	result[index] = resultValue;
}

__global__ void min_plus_kernel_cache_both(int *matrix1, int *matrix2, int *result, int matrixWidth) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = matrixWidth * matrixWidth;

	extern __shared__ int sData[];
	sData[index] = matrix1[index];
	sData[index + offset] = matrix2[index];
	__syncthreads();

	int resultValue = INT_MAX;

	int col = index % matrixWidth;
	int row = index/matrixWidth;
	
	//each thread computes the correct result for a given index in the 2D array
	for (int k = 0; k < matrixWidth; k++) {
		int firstNum = sData[row*matrixWidth + k];
		int secondNum = sData[k*matrixWidth + col + offset];
		//int firstNum = sData[k];
		//int secondNum = matrix2[k*matrixWidth + col];
			
		resultValue = min(resultValue, firstNum + secondNum);
	}
	
	result[index] = resultValue;
}

__global__ void min_plus(int *matrix1, int *matrix2, int *result, int matrixWidth) {
	//printMatrixf("WASSSUP \n");
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int rowNumber = blockIdx.x/(matrixWidth/blockDim.x);
	int firstIndexInRow = rowNumber*matrixWidth;
	
//	int startLoadingAt = matrixWidth*(blockIdx.x


//	int offset = matrixWidth * matrixWidth;
	result[index] = 1;

	extern __shared__ int sData[];
	int numberOfIndicesToLoad = matrixWidth/blockDim.x;
	if (matrixWidth % blockDim.x > threadIdx.x) {
		numberOfIndicesToLoad++;
	}

	for (int i = 0; i < numberOfIndicesToLoad; i++) {
		//sData[index - blockIdx.x*blockDim.x + i*1024] = matrix1[firstIndexInRow + thread + 1024*i];
		sData[threadIdx.x + i*1024] = matrix1[firstIndexInRow + threadIdx.x + 1024*i];
	}
//	sData[index - blockIdx.x*blockDim.x] = matrix1[index];
	//sData[index + offset] = matrix2[index];
	__syncthreads();

//	int col[matrixWidth] = {};
//	for (int i = 0; i < matrixWidth; i++) {
//		col[i] = matrix2[threadIdx.x + i*matrixWidth];
//	}

	int resultValue = INT_MAX;

	int col = index % matrixWidth;
//	int row = index/matrixWidth;

	//each thread computes the correct result for a given index in the 2D array
	for (int k = 0; k < matrixWidth; k++) {
		//int firstNum = sData[row*matrixWidth + k];
		//int secondNum = sData[k*matrixWidth + col + offset];
		int firstNum = sData[k];
		int secondNum = matrix2[k*matrixWidth + col];
		
		resultValue = min(resultValue, firstNum + secondNum);
	}

	result[index] = resultValue;
	//printMatrixf("DONEZO \n");
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

void testHarness (int argc, char *argv[]) {
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

		//int numberOfThreadBlocks = min(matrixWidth, 1024); //ceil(sizeOfMatrix/1024.0);
		//int numberOfThreads = min(matrixWidth, 1024); //min(1024, sizeOfMatrix);
		//int numberOfThreadBlocks = sizeOfMatrix/numberOfThreads;
		//int sharedMemorySize = matrixWidth*sizeof(int); //sizeOfMatrix*2*sizeof(int);

		//execute function

		
		//min_plus<<<numberOfThreadBlocks, numberOfThreads, sharedMemorySize>>>(cudaMatrix1, cudaMatrix2, cudaResult, matrixWidth);
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

		//validate result
		bool matches = true;
		for (int k = 0; k < sizeOfMatrix; k++) {
			if (expected[k] != result[k]) {
				matches = false;
				break;
			}
		}

		cudaFree(cudaMatrix1);
		cudaFree(cudaMatrix2);
		cudaFree(cudaResult);

		if (matches) {
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
	testHarness (argc, argv);
	return 0;
}
