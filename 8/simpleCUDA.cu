//
// simpleCUDA
//
// This simple code sample demonstrates how to perform a simple linear
// algebra operation using CUDA, single precision axpy:
// y[i] = alpha*x[i] + y[i] for x,y in R^N and a scalar alpha
//
// Please refer to the following article for detailed explanations:
// John Nickolls, Ian Buck, Michael Garland and Kevin Skadron
// Scalable parallel programming with CUDA
// ACM Queue, Volume 6 Number 2, pp 44-53, March 2008
// http://mags.acm.org/queue/20080304/
//
// Compilation instructions:
// - Install CUDA
// - Compile with nvcc -o simpleCUDA simpleCUDA.cu
// - Launch the executable
//
// 


/////////////////////////////////////
// standard imports
/////////////////////////////////////
#include <stdio.h>
#include <math.h>

/////////////////////////////////////
// CUDA imports (CUDA runtime, not necessary when compiling with nvcc)
/////////////////////////////////////
//#include <cuda_runtime.h>


/////////////////////////////////////
// global variables and configuration section
/////////////////////////////////////

// problem size (vector length) N
static int N = 123456;

// number of threads per block
static int numThreadsPerBlock = 256;

// device to use in case there is more than one
static int selectedDevice = 0;


/////////////////////////////////////
// kernel function (CPU)
/////////////////////////////////////
void saxpy_serial(int n, float alpha, float *x, float *y)
{
  int i;
  for (i=0; i<n; i++)
    y[i] = alpha*x[i] + y[i];
}


/////////////////////////////////////
// kernel function (CUDA device)
/////////////////////////////////////
__global__ void saxpy_parallel(int n, float alpha, float *x, float *y)
{
  // compute the global index in the vector from
  // the number of the current block, blockIdx,
  // the number of threads per block, blockDim,
  // and the number of the current thread within the block, threadIdx
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // except for special cases, the total number of threads in all blocks
  // adds up to more than the vector length n, so this conditional is
  // EXTREMELY important to avoid writing past the allocated memory for
  // the vector y.
  if (i<n)
    y[i] = alpha*x[i] + y[i];
}


/////////////////////////////////////
// error checking routine
/////////////////////////////////////
void checkErrors(char *label)
{
  // we need to synchronise first to catch errors due to
  // asynchroneous operations that would otherwise
  // potentially go unnoticed

  cudaError_t err;

  err = cudaThreadSynchronize();
  if (err != cudaSuccess)
  {
    char *e = (char*) cudaGetErrorString(err);
    fprintf(stderr, "CUDA Error: %s (at %s)", e, label);
  }

  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    char *e = (char*) cudaGetErrorString(err);
    fprintf(stderr, "CUDA Error: %s (at %s)", e, label);
  }
}


/////////////////////////////////////
// main routine
/////////////////////////////////////
int main (int argc, char **argv)
{
  /////////////////////////////////////
  // (1) initialisations:
  //     - perform basic sanity checks
  //     - set device
  /////////////////////////////////////
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0)
  {
    fprintf(stderr, "Sorry, no CUDA device fount");
    return 1;
  }
  if (selectedDevice >= deviceCount)
  {
    fprintf(stderr, "Choose device ID between 0 and %d\n", deviceCount-1);
    return 1;
  }
  cudaSetDevice(selectedDevice);
  checkErrors("initialisations");
  

  
  /////////////////////////////////////
  // (2) allocate memory on host (main CPU memory) and device,
  //     h_ denotes data residing on the host, d_ on device
  /////////////////////////////////////
  float *h_x = (float*)malloc(N*sizeof(float));
  float *h_y = (float*)malloc(N*sizeof(float));
  float *d_x;
  cudaMalloc((void**)&d_x, N*sizeof(float));
  float *d_y;
  cudaMalloc((void**)&d_y, N*sizeof(float));
  checkErrors("memory allocation");



  /////////////////////////////////////
  // (3) initialise data on the CPU
  /////////////////////////////////////
  int i;
  for (i=0; i<N; i++)
  {
    h_x[i] = 1.0f + i;
    h_y[i] = (float)(N-i+1);
  }



  /////////////////////////////////////
  // (4) copy data to device
  /////////////////////////////////////
  cudaMemcpy(d_x, h_x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, h_y, N*sizeof(float), cudaMemcpyHostToDevice);
  checkErrors("copy data to device");



  /////////////////////////////////////
  // (5) perform computation on host (to enable result comparison later)
  /////////////////////////////////////
  saxpy_serial(N, 2.0f, h_x, h_y);



  /////////////////////////////////////
  // (6) perform computation on device
  //     - we use numThreadsPerBlock threads per block
  //     - the total number of blocks is obtained by rounding the
  //       vector length N up to the next multiple of numThreadsPerBlock
  /////////////////////////////////////
  int numBlocks = (N+numThreadsPerBlock-1) / numThreadsPerBlock;
  saxpy_parallel<<<numBlocks, numThreadsPerBlock>>>(N, 2.0, d_x, d_y);
  checkErrors("compute on device");



  /////////////////////////////////////
  // (7) read back result from device into temp vector
  /////////////////////////////////////
  float *h_z = (float*)malloc(N*sizeof(float));
  cudaMemcpy(h_z, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
  checkErrors("copy data from device");

  
  /////////////////////////////////////
  // (8) perform result comparison
  /////////////////////////////////////
  int errorCount = 0;
  for (i=0; i<N; i++)
  {
    if (abs(h_y[i]-h_z[i]) > 1e-6)
      errorCount = errorCount + 1;
  }
  if (errorCount > 0)
    printf("Result comparison failed.\n");
  else
    printf("Result comparison passed.\n");



  /////////////////////////////////////
  // (9) clean up, free memory
  /////////////////////////////////////
  free(h_x);
  free(h_y);
  free(h_z);
  cudaFree(d_x);
  cudaFree(d_y);
  return 0;
}






