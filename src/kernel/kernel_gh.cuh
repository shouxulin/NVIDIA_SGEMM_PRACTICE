#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

template<const int BLOCK_SIZE>
__global__ void mysgemm_gh(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {

    float *A_start = A; 
    float *B_start = B;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    const int BM = BLOCK_SIZE;
    const int BN = BLOCK_SIZE;
    const int BK = 1;

    int tx = threadIdx.x % BN;
    int ty = threadIdx.x / BN;

    if (bx ==0 && by ==0 && tx ==0 && ty ==0)
    {
        printf("BLOCK_SIZE=%d, BM=%d, BN=%d, BK=%d, gridDim.x=%d, gridDim.y=%d\n", BLOCK_SIZE, BM, BN, BK, gridDim.x, gridDim.y);
    }

    // 申请共享内存空间
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];


    __shared__ float tmps[BLOCK_SIZE];


    bool is_thread_0 = (threadIdx.x == 0 && threadIdx.y == 0 && bx == 0 && by == 0);

    if (is_thread_0)
    {
        // 初始化tmps
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; i++)
        {
            tmps[i] = 0.;
        }
    }
    __syncthreads();



    // 移动到当前block
    // A = &A[by * BM * K];
    // B = &B[bx * BN];
    // C = &C[by * BM * N + bx * BN];

    // A = &A[];
    B = &B[bx * BN];


    // // float block_tmp[BLOCK_SIZE] = {0.};

    // slide across K dimension in B
    #pragma unroll
    for (int k = 0; k < K; k += BK) {

        

        A = &A_start[k * BK];

        if (is_thread_0)
        {
            printf("--- Sliding k=%d, B index = %d\n", k, bx * BN + k * BK * N);
        }
        

        Bs[ty * BN + tx] = B[ty * N + tx];

        __syncthreads();
        if (is_thread_0)
        {
            for (int i = 0; i < BLOCK_SIZE; i++)
            {
                printf("\tBs[%d] = %f ", i, Bs[i]);
            }
            printf("\n");
            
        }


        // slide across M dimension in A
        #pragma unroll
        for (int i = 0; i < M; i += BM)
        {
            if (is_thread_0)
            {
                printf("\t--- Sliding i=%d, A index = %d\n", i, k * BK + i * BM * K);
            }
                
            As[tx * BK] = A[tx * K];

            

            // 同步所有线程缓存完成
            __syncthreads();

            if (is_thread_0)
            {
                for (int i = 0; i < BLOCK_SIZE; i++)
                {
                    printf("\t\tAs[%d] = %f ", i, As[i]);
                }
                printf("\n");
                
            }


            

            #pragma unroll
            for (int l = 0; l < BLOCK_SIZE; l++)
            {
                tmps[i * BM + l] += As[l] * Bs[tx];
            }

            // #pragma unroll
            // for (int l = 0; l < BM; l++)
            // {
            //     tmps[k * BM + l] += block_tmp[l];
            // }
            

            // // FMA计算需要读取缓存数据，在新一轮写入缓存前进行同步，确保所有线程计算完成
            __syncthreads();


            if (is_thread_0)
            {
                for (int l = 0; l < BLOCK_SIZE; l++)
                {
                    printf("\t\ttmps[%d] (C[%d]) = %f ", l, l*N+tx, tmps[i * BM + l]);
                }
                printf("\n");
            }


            A += BM * K;

            // if (is_thread_0)
            // {
            //     printf("\t\tMoving A by BM * K = %d, value=%f\n", BM * K, A[0]);
            // }
            
            

        }

        B += BK * N;
        
    }

    #pragma unroll
    for (int i = 0; i < M; i++)
    {
        C[i * N + tx] = alpha * tmps[i] + beta * C[i * N + tx];
    }






    // float tmp = 0.;
    // for (int k = 0; k < K; k += BK) {
    //     // 缓存A_tile和B_tile --> still enable colascing
    //     As[ty * BK + tx] = A[ty * K + tx];
    //     Bs[ty * BN + tx] = B[ty * N + tx];
    //     // 同步所有线程缓存完成
    //     __syncthreads();
    //     A += BK;
    //     B += BK * N;
    //     for (int i = 0; i < BK; i++) { // --> still enable colascing
    //         tmp += As[ty * BK + i] * Bs[i * BN + tx];
    //     }
    //     // FMA计算需要读取缓存数据，在新一轮写入缓存前进行同步，确保所有线程计算完成
    //     __syncthreads();
    // }
    // C[ty * N + tx] = alpha * tmp + beta * C[ty * N + tx];
}