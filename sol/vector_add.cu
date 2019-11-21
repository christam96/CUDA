#include <cstdio>
#include <iostream>
#include <climits>
#include <algorithm>

using namespace std;

__global__ void min_plus(int *matrix1, int *matrix2, int *result, int matrixWidth, size_t n) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("INDEX: " + index);

	int col = index % matrixWidth;
	int row = index/matrixWidth;

	for (int k = 0; k < matrixWidth; k++) {
		int firstNum = matrix1[row*matrixWidth + k];
		int secondNum = matrix2[k*matrixWidth + col];
		result[index] = min(result[index], firstNum + secondNum);
	}

}


int main() {

	const int matrixWidth = 4;
	const int n = matrixWidth * matrixWidth;    
	
	int matrix1[n] = {0, 62, 51, 77, 66, 0, 9, 96, 37, 53, 0, 60, 83, 25, 16, 0};
	int matrix2[n] = {0, 62, 51, 77, 66, 0, 9, 96, 37, 53, 0, 60, 83, 25, 16, 0};
	int result[n] = {0};



	for (int i = 0; i < n; ++i) {
		result[i] = INT_MAX;
	}

	
	int *matrix1d, *matrix2d, *resultd;
	cudaMalloc((void **)&matrix1d, sizeof(int)*n);
	cudaMalloc((void **)&matrix2d, sizeof(int)*n);
	cudaMalloc((void **)&resultd, sizeof(int)*n);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaMemcpy(matrix1d, matrix1, sizeof(int)*n, cudaMemcpyHostToDevice);
	cudaMemcpy(matrix2d, matrix2, sizeof(int)*n, cudaMemcpyHostToDevice);
	cudaMemcpy(resultd, result, sizeof(int)*n, cudaMemcpyHostToDevice);

	cudaEventRecord(start);

	min_plus<<<1, 16>>>(matrix1d, matrix2d, resultd, matrixWidth, n);

	cudaEventRecord(stop);

	cudaMemcpy(result, resultd, sizeof(int)*n, cudaMemcpyDeviceToHost);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	cout << milliseconds << " milliseconds" << endl;

	for (int i = 0; i < matrixWidth*matrixWidth; i++) {

		if (i % matrixWidth == 0){
			cout << endl;
		}
		cout << result[i] << " ";

	}
	
	cudaFree(matrix1d);
	cudaFree(matrix2d);
	cudaFree(resultd);
	
	return 0;
}
