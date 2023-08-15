#include <time.h>
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

// Assertion to check for errors
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"CUDA_SAFE_CALL: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// #define NUM_THREADS_PER_BLOCK 	256
// #define NUM_BLOCKS 		16
#define PRINT_TIME 		1
#define SM_ARR_LEN		2048
#define TOL			5e-2
#define GIG                     1000000000
// #define CPG                     3.07
// #define IMUL(a, b) __mul24(a, b)
#define BLOCK_SIZE 128
// #define TILE_WIDTH 16
#define SPARSITY 0.05
#define FULL_MASK 0xffffffff

typedef float data_t;

void initializeArray1D(float *arr, int len, float seed);
void initializeSparseMatrixCSR(int *row_offset, int len, int *col_indices, float *values, float seed);

//Md - matrix  
//Nd - vector
//y - result 
// __global__ void MMK(float* Md, float* Nd, float* y)
// {
//     int col = blockDim.x * blockIdx.x + threadIdx.x;
//     int row = blockDim.y * blockIdx.y + threadIdx.y;
//     int num_row = SM_ARR_LEN / BLOCK_SIZE;
//     int k, i;
//     float sum = 0.0f;
//     for (i = 0; i < num_row; i++) {
//         if (col < SM_ARR_LEN || row < SM_ARR_LEN) {
//             for(k = 0; k < BLOCK_SIZE; k++){
//                 sum += Md[row * BLOCK_SIZE + k] * Nd[k];
//             }
//             y[i] = sum;  
//         }
//     }
// }

__device__ float warp_reduce(float val) 
{
    for (int offset = warpSize / 2; offset > 0; offset /= 2) 
        val += __shfl_down_sync (FULL_MASK, val, offset);
    return val;
}

// __global__ void SpMV(int *row_off, float *val, int *col, float *y, float *x)
// {
//     int row = blockDim.y * blockIdx.y + threadIdx.y;
//     int numOfRows = SM_ARR_LEN / BLOCK_SIZE;
//     int i, j;
//     float sum = 0.0f;                                           

//     for (i=0; i < numOfRows; ++i) {
//         if (row < numOfRows) {
//             y[row] = 0.0;
//             for (j=row_off[row]; j<row_off[row+1]; ++j)
//                 sum += val[j] * x[col[j]];
//             y[row] = sum;
//         }   
//     }
// }

__global__ void edge_softmax_forward(int *row_off, float *val, float *y)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int numOfRows = SM_ARR_LEN / BLOCK_SIZE;
    int i, j, k, l;
    float max_score, exp_value, sum;                                         

    for (i=0; i < numOfRows; ++i) {
        if (row < numOfRows) {
            y[row] = 0.0;
            //find max edge value
            for (j=row_off[row]; j<row_off[row+1]; ++j){
                max_score = max(max_score, val[j]);
            }
            //update edge value && find sum of exp
            for (k=row_off[row]; k<row_off[row+1]; ++k) {
                val[k] = val[k] - max_score;
                exp_value = exp(val[k]);
                sum += exp_value;
            }
            
            for (l=row_off[row]; l<row_off[row+1]; ++l) {
                y[row] = exp_value / sum;
            }
        }   
    }
}

__global__ void edge_softmax_forward_warp(int *row_off, float *val, float *y)
{
    int threadId = blockDim.y * blockIdx.y + threadIdx.y;
    int numOfRows = SM_ARR_LEN / BLOCK_SIZE;
    int i, j, k, l;
    float max_score; 
    float exp_value; 
    float sum, res;  
    int warp_id = threadId / 32;
    int lane = threadId % 32; 
    int row = warp_id; //one warp per row

    for (i=0; i < numOfRows; ++i) {
        if (row < numOfRows) {
            y[row] = 0.0;
            for (j=row_off[row]; j<row_off[row+1]; ++j){
                max_score = max(max_score, val[j]);
            }
               
            for (k=row_off[row]; k<row_off[row+1]; ++k) {
                val[k] = val[k] - max_score;
                exp_value = exp(val[k]);
                sum += exp_value;
            }

            //inter communication in warp
            res = warp_reduce(sum);

            if (lane == 0 && row < numOfRows) {
                sum = res;
            }
                
            for (l=row_off[row]; l<row_off[row+1]; ++l) {
                y[row] = exp_value / sum;
            }
        }   
        
    }
}


int compare(float* h_result, float* h_result_gold){
    int i;
    int errCount =0;
    int zeroCount = 0;
    for(i = 0; i < SM_ARR_LEN*SM_ARR_LEN; i++) {
        if (abs(h_result_gold[i] - h_result[i]) > TOL*h_result_gold[i]) {
            errCount++;
        }
        if(h_result[i]==0)
            zeroCount++;
    }
    if (zeroCount>0)
        errCount = -1;
    return errCount;
}

float errorCal(float* h_result, float* h_result_gold){
    int i;
    float error = 0;
    for(i = 0; i < SM_ARR_LEN*SM_ARR_LEN; i++) {
        if(abs(h_result_gold[i] - h_result[i])>error)
            error =  abs(h_result_gold[i] - h_result[i]);
    }
    return error;
}

double interval(struct timespec start, struct timespec end)
{
    struct timespec temp;
    temp.tv_sec = end.tv_sec - start.tv_sec;
    temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    if (temp.tv_nsec < 0) {
        temp.tv_sec = temp.tv_sec - 1;
        temp.tv_nsec = temp.tv_nsec + 1000000000;
    }
    return (((double)temp.tv_sec) + ((double)temp.tv_nsec)*1.0e-9);
}

int main(int argc, char **argv){
    int arrLen = 0;

    // GPU Timing variables
    cudaEvent_t start, stop, start2, stop2;
    float elapsed_gpu;

    // Arrays on GPU global memoryc
    //Md - matrix, Nd - vector, y - result matrix
    float *Md;
    float *Nd;
    float *y;
    int *row_offset;
    int *col_indices;
    float *value;
    float *x;

    // Arrays on the host memory
    float *Md_h;
    float *y_h;
    float *Nd_h;

    if (argc > 1) {
        arrLen  = atoi(argv[1]);
    }
    else {
        arrLen = SM_ARR_LEN * SM_ARR_LEN;
    }


    printf("Length of the array = %d\n", arrLen);

    // Select GPU
    CUDA_SAFE_CALL(cudaSetDevice(0));

    // Allocate GPU memory
    size_t allocSize = arrLen * sizeof(float);
    size_t vectorSize = SM_ARR_LEN * sizeof(float);
    size_t allocSize_int = (SM_ARR_LEN * SM_ARR_LEN) * sizeof(int);
    size_t row_offset_size = (SM_ARR_LEN + 1) * sizeof(int);
    // CUDA_SAFE_CALL(cudaMalloc((void **)&Md, allocSize));
    CUDA_SAFE_CALL(cudaMalloc((void **)&y, vectorSize));
    // CUDA_SAFE_CALL(cudaMalloc((void **)&Nd, vectorSize));
    CUDA_SAFE_CALL(cudaMalloc((void **)&row_offset, row_offset_size));
    // CUDA_SAFE_CALL(cudaMalloc((void **)&col_indices, allocSize_int));
    CUDA_SAFE_CALL(cudaMalloc((void **)&value, allocSize));
    // CUDA_SAFE_CALL(cudaMalloc((void **)&x, allocSize));

    // Allocate arrays on host memory
    y_h		           = (float *) malloc(allocSize);
    // Md_h		           = (float *) malloc(allocSize);
    // Nd_h		           = (float *) malloc(vectorSize);
    int *row_offset_h = (int *)malloc(row_offset_size);
    int *col_indices_h = (int *)malloc(allocSize_int);
    float *values_h = (float *)malloc(allocSize);
    // float *x_h = (float *)malloc(allocSize);


    // Initialize the host arrays
    printf("\nInitializing the arrays ...");
    // Arrays are initialized with a known seed for reproducability
    // initializeArray1D(Md_h, arrLen, 0.53);
    //vector 
    // initializeArray1D(x_h, SM_ARR_LEN, 0.54);
    //sparse matrix
    initializeSparseMatrixCSR(row_offset_h, SM_ARR_LEN, col_indices_h, values_h, 0.54);
    printf("\t... done\n\n");


#if PRINT_TIME
    // Create the cuda events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Record event on the default stream
    cudaEventRecord(start, 0);
#endif

    // Transfer the arrays to the GPU memory
    // CUDA_SAFE_CALL(cudaMemcpy(Md, Md_h, allocSize, cudaMemcpyHostToDevice));
    // CUDA_SAFE_CALL(cudaMemcpy(Nd, Nd_h, vectorSize, cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMemcpy(row_offset, row_offset_h, row_offset_size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(value, values_h, allocSize, cudaMemcpyHostToDevice));
    // CUDA_SAFE_CALL(cudaMemcpy(col_indices, col_indices_h, allocSize_int, cudaMemcpyHostToDevice));
    // CUDA_SAFE_CALL(cudaMemcpy(x, x_h, allocSize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(y, y_h, allocSize, cudaMemcpyHostToDevice));


    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2, 0);

    dim3 dimGrid(SM_ARR_LEN, SM_ARR_LEN);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    // Launch the kernel
    edge_softmax_forward_warp<<<dimGrid, dimBlock>>>(row_offset, value, y);

    // timer for kernel execution
    cudaEventRecord(stop2,0);
    cudaEventSynchronize(stop2);
    cudaEventElapsedTime(&elapsed_gpu, start2, stop2);
    printf("\nGPU kernel execution time: %f (msec)\n", elapsed_gpu);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);

    // Check for errors during launch
    CUDA_SAFE_CALL(cudaPeekAtLastError());

    // Transfer the results back to the host
    CUDA_SAFE_CALL(cudaMemcpy(y_h, y, allocSize, cudaMemcpyDeviceToHost));
    // CUDA_SAFE_CALL(cudaMemcpy(col_indices_h, col_indices, allocSize_int, cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(row_offset_h, row_offset, row_offset_size, cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(values_h, value, allocSize, cudaMemcpyDeviceToHost));
    // CUDA_SAFE_CALL(cudaMemcpy(x_h, x, allocSize, cudaMemcpyDeviceToHost));

#if PRINT_TIME
    // Stop and destroy the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_gpu, start, stop);
    printf("\nGPU start-to-finish time: %f (msec)\n", elapsed_gpu);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
#endif

    // printf("\nCompare: %d\n\n\n",compare(y_h,y_h_gold));
    // printf("\nBiggest Error: %f\n\n\n",errorCal(y_h,y_h_gold));

    // Free-up device and host memory
    CUDA_SAFE_CALL(cudaFree(y));
    CUDA_SAFE_CALL(cudaFree(value));
    // CUDA_SAFE_CALL(cudaFree(x));
    // CUDA_SAFE_CALL(cudaFree(col_indices));
    CUDA_SAFE_CALL(cudaFree(row_offset));


    free(y_h);
    free(values_h);
    // free(col_indices_h);
    free(row_offset_h);
    // free(x_h);

    return 0;
}

struct timespec diff(struct timespec start, struct timespec end)
{
    struct timespec temp;
    if ((end.tv_nsec-start.tv_nsec)<0) {
        temp.tv_sec = end.tv_sec-start.tv_sec-1;
        temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
    } else {
        temp.tv_sec = end.tv_sec-start.tv_sec;
        temp.tv_nsec = end.tv_nsec-start.tv_nsec;
    }
    return temp;
}


void initializeArray1D(float *arr, int len, float seed) {
    int i;
    float randNum;
    srand(seed);

    for (i = 0; i < len; i++) {
        randNum = (float) (rand() / 100000);
        arr[i] = randNum;
    }
}

void initializeSparseMatrixCSR(int *row_offset, int len, int *col_indices, float *values, float seed) {
    //num of non-zero elements
    int nnz = 0;
    int i, j;
   
    srand(seed);

    for (i = 0; i < len; ++i) {
        // row_offset[i] = row_offset[i - 1];
        for (int j = 0; j < len; ++j) {
            if ((float)rand() / RAND_MAX < SPARSITY) {
                col_indices[nnz] = j;
                values[nnz] = (float)rand() / RAND_MAX;
                nnz++;
                // row_offset[nnz]++;
            }
        }
        row_offset[i + 1] = nnz;
    }
}

