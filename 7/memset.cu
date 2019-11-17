#include <iostream>
#include <ctime>
#include <cassert>
#include <stdio.h>

#define start_timer(id) \
    cudaEvent_t start##id, stop##id; \
    cudaEventCreate(&start##id); \
    cudaEventCreate(&stop##id); \
    cudaEventRecord(start##id, 0)

#define stop_timer(id, elapsedTime) \
    cudaEventRecord(stop##id, 0); \
    cudaEventSynchronize(stop##id); \
    cudaEventElapsedTime(&elapsedTime, start##id, stop##id); \
    cudaEventDestroy(start##id); \
    cudaEventDestroy(stop##id)


using namespace std;

static inline 
void cpu_memory_bandwidth(int *X, int n, int m = 1) {
    for (int i = 0; i < m; ++i)
        memset(X, 0, n*sizeof(int));
}

static inline
void cpu_gpu_memory(int *X_d, int *X, int n, int m = 1) {
    int size = sizeof(int)*n;
    for (int i = 0; i < m; ++i)
        cudaMemcpy(X_d, X, size, cudaMemcpyHostToDevice);
}

static inline
void gpu_cpu_memory(int *X, int *X_d, int n, int m = 1) {
    int size = sizeof(int)*n;
    for (int i = 0; i < m; ++i)
        cudaMemcpy(X, X_d, size, cudaMemcpyDeviceToHost);
}

void gpu_gpu_memory(int *Y_d, int *X_d, int n, int m = 1) {
    int size = sizeof(int)*n;
    for (int i = 0; i < m; ++i)
        cudaMemcpy(Y_d, X_d, size, cudaMemcpyDeviceToDevice);
}

#define E_THREAD 8
#define N_THREAD (1<<E_THREAD)

__global__
void reset(int *X, int m) {
    int bid = (blockIdx.y << 15) + blockIdx.x;
    int tid = (bid << E_THREAD) + threadIdx.x; 
    for (int i = 0; i < m; ++i) {
        X[tid] = 0;
    }
}

void gpu_kernel(int *X_d, int n, int m) {
    int nb = (n >> E_THREAD);
    dim3 nBlock(nb);
    if (nb > (1 << 15)) {
        nBlock.x = (1 << 15);
        nBlock.y = (nb >> 15);
    }
    reset<<<nBlock, N_THREAD>>>(X_d, m);
}

int main(int argc, char** argv) {
    int m = 1, e = 28, n = (1L << e);
    if (argc > 1) {
        e = atoi(argv[1]);
        n = (1L << e);
        assert(n > 0);
    }
    int *X = new int[n];
    int *X_d;
    int *Y_d;
    cudaMalloc((void **)&X_d, n*sizeof(int));
    cudaMalloc((void **)&Y_d, n*sizeof(int));

    clock_t t1, t2;
    double dt;

    // CPU - Main Memory
    t1 = clock();
    cpu_memory_bandwidth(X, n, m);
    t2 = clock();
    dt = (t2 - t1) / (double)CLOCKS_PER_SEC;
    printf("CPU -> CPU :: time %.4lf\t", dt);
    printf("bandwidth %.1lf MB/s\n", sizeof(int)*(n >> 20) / dt);
    

    // CPU -> GPU  
    t1 = clock();
    cpu_gpu_memory(X_d, X, n, m);
    t2 = clock();
    dt = (t2 - t1) / (double)CLOCKS_PER_SEC;
    printf("CPU -> GPU :: time %.4lf\t", dt);
    printf("bandwidth %.1lf MB/s\n", sizeof(int)*(n >> 20) / dt);

    // GPU -> CPU  
    t1 = clock();
    gpu_cpu_memory(X, X_d, n, m);
    t2 = clock();
    dt = (t2 - t1) / (double)CLOCKS_PER_SEC;
    printf("GPU -> CPU :: time %.4lf\t", dt);
    printf("bandwidth %.1lf MB/s\n", sizeof(int)*(n >> 20) / dt);
    
    // GPU -> GPU  
    float elapsedTime;
    start_timer(0);
    for (int i = 0; i < m; ++i) cudaMemcpy(Y_d, X_d, n*sizeof(int), cudaMemcpyDeviceToDevice);
    stop_timer(0, elapsedTime);
    printf("GPU -> GPU :: time %.4lf\t", elapsedTime / 1000);
    printf("bandwidth %.1lf GB/s\n", sizeof(int)*((double)n / (1 << 30)) / (elapsedTime / 1000));

    // GPU kernel
    start_timer(1);
    gpu_kernel(X_d, n, m);
    stop_timer(1, elapsedTime);
    printf("GPU kernel :: time %.4lf\t", elapsedTime / 1000);
    printf("bandwidth %.1lf GB/s\n", sizeof(int)*((double)n / (1 << 30)) / (elapsedTime / 1000));

    delete [] X;
    cudaFree(X_d);
    cudaFree(Y_d);

    return 0;
}
