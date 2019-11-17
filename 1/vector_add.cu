#include <cstdio>

using namespace std;

__global__ void vector_addition(int *A, int *B, size_t n) {
	A[threadIdx.x] += B[threadIdx.x];
}

int main() {
	const int n = 128;    int A[n] = {0};
	int B[n] = {0};

	for (int i = 0; i < n; ++i) A[i] = i, B[i] = n - i;
	
	int *Ad, *Bd;
	cudaMalloc((void **)&Ad, sizeof(int)*n);
	cudaMalloc((void **)&Bd, sizeof(int)*n);

	cudaMemcpy(Ad, A, sizeof(int)*n, cudaMemcpyHostToDevice);
	cudaMemcpy(Bd, B, sizeof(int)*n, cudaMemcpyHostToDevice);

	vector_addition<<<1, n>>>(Ad, Bd, n);
	cudaMemcpy(A, Ad, sizeof(int)*n, cudaMemcpyDeviceToHost);

	for (int i = 0; i < n; ++i) printf("%2d ", A[i]);
	printf("\n");
	
	cudaFree(Ad);
	cudaFree(Bd);
	
	return 0;
}
