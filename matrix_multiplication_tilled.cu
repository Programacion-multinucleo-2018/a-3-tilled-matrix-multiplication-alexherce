#include "common.h"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

using namespace std;

#define MATRIXSIZE 2000
#define TILESIZE 32

// Multiply Matrices in GPU
__global__ void multMatrixGPU(float *MatA, float *MatB, float *MatC, int nx, int ny)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x; // col
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y; // row
	unsigned int idx = iy * nx + ix;

	float sum = 0;
	if (ix < nx && iy < ny) {
		for (int i = 0; i < nx; i++) {
			sum += MatA[iy * nx + i] * MatB[i * ny + ix];
		}
		MatC[idx] = sum;
	}
}

// Multiply Matrices in GPU with Tiles
__global__ void multMatrixGPUTiles(float *MatA, float *MatB, float *MatC, int nx, int ny)
{
	unsigned int ix = threadIdx.x + TILESIZE * blockIdx.x;
	unsigned int iy = threadIdx.y + TILESIZE * blockIdx.y;
	float sum = 0.0;

	__shared__ float TempA[TILESIZE][TILESIZE];
	__shared__ float TempB[TILESIZE][TILESIZE];

	TempA[threadIdx.y][threadIdx.x] = 0.0;
	TempB[threadIdx.y][threadIdx.x] = 0.0;

	__syncthreads();

	for (int i = (TILESIZE + nx - 1) / TILESIZE; i >= 0; i--)
	{
		if ((i * TILESIZE + threadIdx.x < nx) && (iy < ny)) {
			TempA[threadIdx.y][threadIdx.x] = MatA[iy * ny + i * TILESIZE + threadIdx.x];
		}

		if ((i * TILESIZE + threadIdx.y < ny) && (ix < nx)) {
			TempB[threadIdx.y][threadIdx.x] = MatB[(i * TILESIZE + threadIdx.y) * ny + ix];
		}

		__syncthreads();

		for (int j = 0; j < TILESIZE; j++) {
			sum += TempA[threadIdx.y][j] * TempB[j][threadIdx.x];
		}

		__syncthreads();
	}

	if (ix < nx && iy < ny)
	{
		MatC[iy * ny + ix] = sum;
	}
}

// Multiply Matrices in CPU
void multMatrixCPU(float *A, float *B, float *C, const int nx, const int ny)
{
	float *ia = A;
	float *ib = B;
	float *ic = C;
	float sum = 0;

	for (int i = 0; i < nx; i++) {
		for (int j = 0; j < nx; j++) {
			for (int k = 0; k < nx; k++) {
				ic[i * nx + j] += ia[i * nx + k] * ib[j + k * nx];
			}
		}
	}

	return;
}

// Multiply Matrix in CPU Parallel
void multMatrixCPUParallel(float *A, float *B, float *C, const int nx, const int ny)
{
	float *ia = A;
	float *ib = B;
	float *ic = C;
	float sum = 0;

	int i, j, k;

	int nProcessors = omp_get_max_threads();

	std::cout << "CPU processors available: " << nProcessors << std::endl;

	omp_set_num_threads(6);

#pragma omp parallel for private(sum,i,j,k) shared(ia, ib, ic)
	for (i = 0; i < nx; i++) {
		for (j = 0; j < nx; j++) {
			sum = 0;
			for (k = 0; k < nx; k++) {
				sum += ia[i * nx + k] * ib[k * nx + j];
			}
			ic[i * nx + j] = sum;
		}
	}

	return;
}

void initialData(float * ip, const int size)
{
	int i;
	for (i = 0; i < size; i++)
	{
		ip[i] = (float)rand() / (RAND_MAX / 10.0f);
	}

	return;
}

void checkResult(float *hostRef, float *gpuRef, const int nxy)
{
	double epsilon = 0.5;
	bool match = 1;

	for (int i = 0; i < nxy; i++)
	{
		if (abs(hostRef[i] - gpuRef[i]) > epsilon)
		{
			match = 0;
			printf("host %f gpu %f\n", hostRef[i], gpuRef[i]);
			break;
		}
	}

	if (match)
		printf("Arrays match.\n\n");
	else
		printf("Arrays do not match.\n\n");
}

int main(int argc, char **argv)
{
	printf("%s Starting...\n", argv[0]);

	// set up device
	int dev = 0;
	cudaDeviceProp deviceProp;
	SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev), "Error device prop");
	printf("Using Device %d: %s\n", dev, deviceProp.name);
	SAFE_CALL(cudaSetDevice(dev), "Error setting device");

	// set up data size of matrix
	int nx = MATRIXSIZE;
	int ny = MATRIXSIZE;

	int nxy = nx * ny;
	int nBytes = nxy * sizeof(float);

	printf("Matrix size: nx %d ny %d\n", nx, ny);
	printf("Tile size: %d\n\n", TILESIZE);

	// malloc host memory
	float *h_A , *h_B , *hostRef, *gpuRef, *gpuRefTiles;
	h_A  = (float *)malloc(nBytes);
	h_B  = (float *)malloc(nBytes);
	hostRef = (float *)malloc(nBytes);
	gpuRef = (float *)malloc(nBytes);
	gpuRefTiles = (float *)malloc(nBytes);

	// initialize data at host side
	initialData(h_A , nxy);
	initialData(h_B , nxy);

	memset(hostRef, 0, nBytes);
	memset(gpuRefTiles, 0, nBytes);
	memset(gpuRef, 0, nBytes);

	// ------------------- CPU -------------------
	auto start_cpu = chrono::high_resolution_clock::now();
	multMatrixCPUParallel(h_A , h_B , hostRef, nx, ny);
	auto end_cpu = chrono::high_resolution_clock::now();
	chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;

	printf("multMatrixCPUParallel elapsed %f ms\n\n", duration_ms.count());

	// ------------------- GPU Setup -------------------
	// malloc device global memory
	float *d_MatA, *d_MatB, *d_MatC;
	SAFE_CALL(cudaMalloc((void **)&d_MatA, nBytes), "Error allocating d_MatA");
	SAFE_CALL(cudaMalloc((void **)&d_MatB, nBytes), "Error allocating d_MatB");
	SAFE_CALL(cudaMalloc((void **)&d_MatC, nBytes), "Error allocating d_MatC");

	// transfer data from host to device
	SAFE_CALL(cudaMemcpy(d_MatA, h_A , nBytes, cudaMemcpyHostToDevice), "Error copying d_MatA");
	SAFE_CALL(cudaMemcpy(d_MatB, h_B , nBytes, cudaMemcpyHostToDevice), "Error copying d_MatB");

	// invoke kernel at host side
	dim3 block(TILESIZE, TILESIZE);
	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

	// ------------------- GPU Normal -------------------
	start_cpu = chrono::high_resolution_clock::now();
	multMatrixGPU << <grid, block >> > (d_MatA, d_MatB, d_MatC, nx, ny);
	SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel");
	end_cpu = chrono::high_resolution_clock::now();
	duration_ms = end_cpu - start_cpu;


	printf("multMatrixGPU <<<(%d,%d), (%d,%d)>>> elapsed %f ms\n", grid.x, grid.y, block.x, block.y, duration_ms.count());

	// SAFE_CALL kernel error
	SAFE_CALL(cudaGetLastError(), "Error with last error");

	// copy kernel result back to host side
	SAFE_CALL(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost), "Error copying d_MatC");

	checkResult(hostRef, gpuRef, nxy);

	// ------------------- GPU TILES -------------------
	start_cpu = chrono::high_resolution_clock::now();
	multMatrixGPUTiles << <grid, block >> > (d_MatA, d_MatB, d_MatC, nx, ny);
	SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel");
	end_cpu = chrono::high_resolution_clock::now();
	duration_ms = end_cpu - start_cpu;

	printf("multMatrixGPUTiles <<<(%d,%d), (%d,%d)>>> elapsed %f ms\n", grid.x, grid.y, block.x, block.y, duration_ms.count());

	// SAFE_CALL kernel error
	SAFE_CALL(cudaGetLastError(), "Error with last error");

	// copy kernel result back to host side
	SAFE_CALL(cudaMemcpy(gpuRefTiles, d_MatC, nBytes, cudaMemcpyDeviceToHost), "Error copying d_MatC");

	checkResult(hostRef, gpuRefTiles, nxy);

	// free device global memory
	SAFE_CALL(cudaFree(d_MatA), "Error freeing memory");
	SAFE_CALL(cudaFree(d_MatB), "Error freeing memory");
	SAFE_CALL(cudaFree(d_MatC), "Error freeing memory");

	// free host memory
	free(h_A );
	free(h_B );
	free(hostRef);
	free(gpuRef);

	// reset device
	SAFE_CALL(cudaDeviceReset(), "Error reseting");

	return (0);
}
