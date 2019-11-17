#include <iostream>
#include <stdio.h>

/**
 * Given a device array of integers, compute the index of first nonzero
 * entry in the array, from left to right. 
 *
 * For example, opearting on the array
 *
 *  0  1  2   3  4  5  6
 * [0, 0, 0, -1, 0, 0, 2] 
 *
 * gets the index 3. The result is stored into deg_ptr (initial value is n).
 *
 */
template <int N_THD>
__global__ void degree_ker(const int *X, int n, int* deg_ptr) {
    int tid = blockIdx.x * N_THD + threadIdx.x;    
    
    if ((tid < n) && (X[tid] != 0)) {
        atomicMin(deg_ptr, tid);    
    }
}

using namespace std;

int main(int argc, char** argv) {
    int n = 30;
    if (argc > 1) n = atoi(argv[1]);

    int *X = new int[n+1]();

    srand(time(NULL));
    int r = rand() % n + 1;
    for (int i = 0; i < n; ++i) { X[i] = i / r; }
    X[n] = n;
 
    //for (int i = 0; i <= n; ++i) printf("%2d ", i);
    //printf("\n");
    //for (int i = 0; i <= n; ++i) printf("%2d ", X[i]);
    //printf("\n");

    int *X_d;
    cudaMalloc((void **)&X_d, sizeof(int)*(n+1));
    cudaMemcpy(X_d, X, sizeof(int)*(n+1), cudaMemcpyHostToDevice);
    
    const int nthd = 16;
    int nb = (n / nthd) + ((n % nthd) ? 1 : 0);
    
    int *deg_dev = X_d + n;

    degree_ker<nthd><<<nb, nthd>>>(X_d, n, deg_dev);

    int deg;
    cudaMemcpy(&deg, deg_dev, sizeof(int), cudaMemcpyDeviceToHost);
    printf("r = %d, index = %d\n", r, deg);

    delete [] X;
    cudaFree(X_d);

    return 0;  
}
