#include <time.h>
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <omp.h>

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
#define BLOCK_SIZE 16
// #define TILE_WIDTH 16

typedef float data_t;

void initializeArray1D(float *arr, int len, float seed);

//Md - matrix  
//Nd - vector
//Pd - result 
__global__ void MMK(float* Md, float* Nd, float* Pd)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int num_row = SM_ARR_LEN / BLOCK_SIZE;
    int k, i;
    float sum = 0.0f;
    for (i = 0; i < num_row; i++) {
        if (col < SM_ARR_LEN || row < SM_ARR_LEN) {
            for(k = 0; k < BLOCK_SIZE; k++){
                sum += Md[row * BLOCK_SIZE + k] * Nd[k];
            }
            Pd[i] = sum;  
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
    //Md - matrix, Nd - vector, Pd - result matrix
    float *Md;
    float *Nd;
    float *Pd;

    // Arrays on the host memory
    float *Md_h;
    float *Pd_h;
    float *Nd_h;
    // float *Pd_h_gold;
    // float *Pd_h_cpu_block;
    // int i, errCount = 0, zeroCount = 0;

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
    CUDA_SAFE_CALL(cudaMalloc((void **)&Md, allocSize));
    CUDA_SAFE_CALL(cudaMalloc((void **)&Pd, vectorSize));
    CUDA_SAFE_CALL(cudaMalloc((void **)&Nd, vectorSize));

    // Allocate arrays on host memory
    Pd_h		           = (float *) malloc(vectorSize);
    Md_h		           = (float *) malloc(allocSize);
    Nd_h		           = (float *) malloc(vectorSize);

    // Initialize the host arrays
    printf("\nInitializing the arrays ...");
    // Arrays are initialized with a known seed for reproducability
    initializeArray1D(Md_h, arrLen, 0.53);
    initializeArray1D(Nd_h, SM_ARR_LEN, 0.54);
    printf("\t... done\n\n");


    struct timespec diff(struct timespec start, struct timespec end);
    struct timespec time1, time2;
    struct timespec time_stamp;


#if PRINT_TIME
    // Create the cuda events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Record event on the default stream
    cudaEventRecord(start, 0);
#endif

    // Transfer the arrays to the GPU memory
    CUDA_SAFE_CALL(cudaMemcpy(Md, Md_h, allocSize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(Nd, Nd_h, vectorSize, cudaMemcpyHostToDevice));
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2, 0);

    dim3 dimGrid(SM_ARR_LEN, SM_ARR_LEN);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    // Launch the kernel
    //Md - matrix, Nd - vector, Pd - result 
    MMK<<<dimGrid, dimBlock>>>(Md, Nd, Pd);

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
    CUDA_SAFE_CALL(cudaMemcpy(Pd_h, Pd, vectorSize, cudaMemcpyDeviceToHost));

#if PRINT_TIME
    // Stop and destroy the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_gpu, start, stop);
    printf("\nGPU start-to-finish time: %f (msec)\n", elapsed_gpu);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
#endif

    // printf("\nCompare: %d\n\n\n",compare(Pd_h,Pd_h_gold));
    // printf("\nBiggest Error: %f\n\n\n",errorCal(Pd_h,Pd_h_gold));

    // Free-up device and host memory
    CUDA_SAFE_CALL(cudaFree(Pd));
    CUDA_SAFE_CALL(cudaFree(Md));
    CUDA_SAFE_CALL(cudaFree(Nd));


    free(Pd_h);
    free(Md_h);
    free(Nd_h);
    // free(Pd_h_gold);

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