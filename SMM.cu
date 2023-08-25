#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include "kernel.h"
#include "invoke.h"
#include "op.h"
#include <cassert>
#include <iostream>
#include <limits>
#include "wtime.h"

//warp per row (best)
__global__ void 
spmme_warp (const coo_t* __restrict__  obj1, const float* __restrict__ edge_weight, float* __restrict__ y,
            op_t op, const float init_value, int dim) 
{

    op_scalar_fn op_fn=get_fn_kernel(op);
    
    // Thread ID within warp
    int t = threadIdx.x;
    int id = t & (WARP_SIZE-1);
    int warpsPerBlock = blockDim.x / WARP_SIZE;
    
    // One row per warp
    int myRow = (blockIdx.x * warpsPerBlock) + (t / WARP_SIZE);
    
    if(myRow >= obj1->v_count) {
        return; 
    }
    
    float  message = init_value;
    int start = obj1 -> csr_offset[myRow];
    int end   = obj1 -> csr_offset[myRow +  1]; 
    
    for (int i = start + id ; i < end; i+=WARP_SIZE){
        message = op_fn(message, edge_weight[i]);
    } 
    message = warp_reduce_op(message, op_fn);
    if (id == 0) {
        y[myRow] = message;
    } 
}                       

__global__ void 
spmmeid_warp (const coo_t* __restrict__  obj1, const float* __restrict__ edge_weight, 
                float* __restrict__ y, op_t op, const float init_value, int dim) 
{

    op_scalar_fn op_fn=get_fn_kernel(op);
    
    // Thread ID within warp
    int t = threadIdx.x;
    int id = t & (WARP_SIZE-1);
    int warpsPerBlock = blockDim.x / WARP_SIZE;
    
    // One row per warp
    int myRow = (blockIdx.x * warpsPerBlock) + (t / WARP_SIZE);
    
    if(myRow >= obj1->v_count) {
        return; 
    }
    
    float  message = init_value;
    int start = obj1 -> csr_offset[myRow];
    int end   = obj1 -> csr_offset[myRow +  1];
    int eid   = 0;
    
    for (int i = start + id ; i < end; i+=WARP_SIZE) {
        message = op_fn(message, edge_weight[obj1->csr_eid[i]]);
    } 
    message = warp_reduce_op(message, op_fn);
    if (id == 0) {
        y[myRow] = message;
    } 
}                       


__global__ void  spmme2d_warp(const coo_t* __restrict__  obj1, const float* __restrict__ edge_weight, 
                         float* y, op_t op, float init_value, int output_dim) 
{
    /*
    int t = threadIdx.x;
    op_scalar_fn op_fn=get_fn_kernel(op);
    // Thread ID within warp
    int id = t & (WARP_SIZE-1);
    int warpsPerBlock = blockDim.x / WARP_SIZE;
    // One row per warp
    int myRow = (blockIdx.x * warpsPerBlock) + (t / WARP_SIZE);
    __shared__ volatile float partialSums[BLOCK_SIZE];
    if(myRow >= obj1->v) {
        return; 
    }

    vid_t * nebrs;
    vid_t eid = 0;
    
    const float* edge_weight_per_edge;
    float* message_reduce = y+output_dim*myRow;

    int degree = obj1 -> get_nebrs(myRow, nebrs);
    if (degree == 0){
        for (int64_t h = 0; h < output_dim; h++) {
            message_reduce[h] = 0;
        }
        return;
    }
    
    for (int64_t h = 0; h < output_dim; h++) {
        float  message=init_value;
        for (int i = id; i < degree; i+=WARP_SIZE) {
            eid = obj1 -> get_eid(nebrs, i);
            edge_weight_per_edge = edge_weight+output_dim*eid;
            message = op_fn(message, edge_weight_per_edge[h]);
        }
        message = warp_reduce_op(message, op_fn);
        if (id == 0) {
            message_reduce[h] = message;
        }
    }*/
}

__global__ void 
edge_softmax_forward (const coo_t* __restrict__ obj1, const float* __restrict__ edge_weight, 
    float* __restrict__ output, bool reverse, int dim) 
{
    int myRow = blockDim.x * blockIdx.x + threadIdx.x;
    float max_score = -INFINITY, exp_sum = 0.0f; 
    
    if (myRow >= obj1 -> v_count) {
        return;
    }
      
    int start = obj1 -> csr_offset[myRow];
    int end   = obj1 -> csr_offset[myRow + 1];
    
    //find max edge value for the row
    for (int i = start; i < end; ++i) {
        max_score = fmaxf(max_score, edge_weight[i]);
    }
    //update edge value && find exp_sum for the row
    for (int k = start; k < end; ++k) {
        output[k] = expf(edge_weight[k] - max_score);
        exp_sum += output[k];
    }
    //output non zero element for the row
    for (int l = start; l < end; ++l) {
        output[l] = output[l] / exp_sum;
    }
} 

__global__ void 
edge_softmax_forward_warp (const coo_t* __restrict__ obj1, const float* __restrict__ edge_weight, 
    float* __restrict__ output, bool reverse, int dim) 
{
    int t = threadIdx.x;
    int id = t & (WARP_SIZE-1);
    int warpsPerBlock = blockDim.x / WARP_SIZE;
    
    // One row per warp
    int myRow = (blockIdx.x * warpsPerBlock) + (t / WARP_SIZE);
    
    float max_score = -INFINITY;
    float exp_sum = 0.0f; 
      
    if (myRow >= obj1 -> v_count) {
        return;
    }

    int start = obj1 -> csr_offset[myRow];
    int end   = obj1 -> csr_offset[myRow + 1];
    
    //find max edge value for the row
    for (int i = start; i < end; i += WARP_SIZE) {
        max_score = fmaxf(max_score, edge_weight[i]);
    }

    // find the max value among max_score across all threads in warp
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        max_score = fmaxf(max_score, __shfl_down_sync(FULL_WARP_MASK, max_score, offset));
    }
    // broadcast max value in the warp from thread 0
    max_score = __shfl_sync(FULL_WARP_MASK, max_score, 0, WARP_SIZE);

    for (int k = start; k<end; k += WARP_SIZE) {
        output[k] = expf(edge_weight[k] - max_score);
        exp_sum += output[k];
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        exp_sum += __shfl_down_sync(FULL_WARP_MASK, exp_sum, offset);
    }

    exp_sum = __shfl_sync(FULL_WARP_MASK, exp_sum, 0, WARP_SIZE);

    //normalize edge values using exponential sum
    for (int l = start; l < end; l += WARP_SIZE) {
        output[l] = output[l] / exp_sum;
    }
}   

__global__ void 
edge_softmax_backward (const coo_t* __restrict__ obj1, float* __restrict__ dZ, float* __restrict__ out, 
            float* __restrict__ output, bool reverse, int dim) 
{
    int myRow = blockDim.x * blockIdx.x + threadIdx.x;
    float accum = 0.0f; 
    
    if (myRow >= obj1 -> v_count) {
        return;
    }

    int start = obj1 -> csr_offset[myRow];
    int end   = obj1 -> csr_offset[myRow + 1];
    
    for (int i = start; i < end; ++i){
        accum += out[i] * dZ[i];
    }
    
    for (int j=start; j<end; ++j) {
        output[j] = (out[j] * dZ[j]) - (out[j] * accum);
    }
}   

__global__ void 
edge_softmax_backward_warp(const coo_t* __restrict__ obj1, float* __restrict__ dZ, float* __restrict__ out, 
    float* __restrict__ output, bool reverse, int dim) 
{
    int t = threadIdx.x;
    int id = t & (WARP_SIZE-1);
    int warpsPerBlock = blockDim.x / WARP_SIZE;
    
    // One row per warp
    int myRow = (blockIdx.x * warpsPerBlock) + (t / WARP_SIZE);
    float accum = 0.0f; 
    
    if (myRow >= obj1 -> v_count) {
        return;
    }

    int start = obj1 -> csr_offset[myRow];
    int end   = obj1 -> csr_offset[myRow + 1];
    
    //find max edge value for the row
    for (int i = start; i < end; i += WARP_SIZE){
        accum += out[i] * dZ[i];
    }
    //warp reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        accum += __shfl_down_sync(FULL_WARP_MASK, accum, offset);
    }
    //broadcast
    accum = __shfl_sync(FULL_WARP_MASK, accum, 0, WARP_SIZE); 
    
    for (int j = start; j < end; j += WARP_SIZE) {
        output[j] = (out[j] * dZ[j]) - (out[j] * accum);
    }
}

/****** INVOKE START HERE *********/
void invoke_spmme(coo_t * obj1, float* edge_weight, float* y, op_t op, bool reverse, int total_dim) 
{
    float init_value = 0;
    if (op == eMAX) init_value = -INFINITY;
    //if (op == eMAX) init_value = -999.9;
    else if (op == eMIN) init_value = INFINITY;
    //else if (op == eMIN) init_value = 999.9;
    else init_value = 0;
    
    int nBlocks =  (obj1->v_count + WARP_PER_BLOCK - 1) / WARP_PER_BLOCK;
    spmme_warp <<<nBlocks,BLOCK_SIZE>>> (obj1, edge_weight, y, op, init_value, total_dim);
    
    //cudaDeviceSynchronize();
}

void invoke_spmmeid(coo_t * obj1, float* edge_weight, float* y, op_t op, bool reverse, int total_dim) 
{
    float init_value = 0;
    if (op == eMAX) init_value = -INFINITY;
    //if (op == eMAX) init_value = -999.9;
    else if (op == eMIN) init_value = INFINITY;
    //else if (op == eMIN) init_value = 999.9;
    else init_value = 0;
    
    int nBlocks =  (obj1->v_count + WARP_PER_BLOCK - 1) / WARP_PER_BLOCK;
    spmmeid_warp <<<nBlocks,BLOCK_SIZE>>> (obj1, edge_weight, y, op, init_value, total_dim);
    
    //cudaDeviceSynchronize();
}

void invoke_edge_softmax_forward(coo_t * obj1, float* edge_weight, float* output, bool reverse, int total_dim) {
    //printf("invoke edge_softmax_forward start\n");
    // int nBlocks = (obj1->v_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // edge_softmax_forward<<<nBlocks,BLOCK_SIZE >>> (obj1, edge_weight, output, reverse, total_dim);

    int nBlocks = (obj1->v_count + WARP_PER_BLOCK - 1) / WARP_PER_BLOCK;
    edge_softmax_forward_warp<<<nBlocks, BLOCK_SIZE >>> (obj1, edge_weight, output, reverse, total_dim);
    //printf("invoke edge_softmax_forward end\n");
}

void invoke_edge_softmax_backward(coo_t * obj1, float* edge_weight, float* out, float* output, bool reverse, int total_dim) {
    //printf("invoke edge_softmax_backward start\n");
    // int nBlocks = (obj1->v_count + BLOCK_SIZE -1) / BLOCK_SIZE;
    // edge_softmax_backward<<<nBlocks,BLOCK_SIZE >>> (obj1, edge_weight, out, output, reverse, total_dim);

    int nBlocks = (obj1->v_count + WARP_PER_BLOCK - 1) / WARP_PER_BLOCK;
    edge_softmax_backward_warp<<<nBlocks, BLOCK_SIZE >>> (obj1, edge_weight, out, output, reverse, total_dim);

    //printf("invoke edge_softmax_forward end\n");
}


