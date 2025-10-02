#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define CEIL_DIV(M, N) ((M) + (N)-1) / (N)

template<const int BLOCK_DIM>
__global__ void mysgemm_gh(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {

    float *A_start = A; 
    // float *B_start = B;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    const int BM = BLOCK_DIM;
    const int BN = BLOCK_DIM;
    const int BK = BLOCK_DIM;

    // int tx = threadIdx.x % BN;
    // int ty = threadIdx.x / BN;
    int tx = threadIdx.x;
    int ty = threadIdx.y;


    // if (threadIdx.x==0 && threadIdx.y==0)
    // {
    //     printf("BLOCK_DIM=%d, BM=%d, BN=%d, BK=%d, gridDim.x=%d, gridDim.y=%d, blockIdx.x=%d, blockIdx.y=%d\n", BLOCK_DIM, BM, BN, BK, gridDim.x, gridDim.y, bx, by);
    // }




    // 申请共享内存空间
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];


    // bool is_thread_0 = (threadIdx.x == 0 && threadIdx.y == 0 && bx == 0 && by == 0);
    // bool is_thread_0 = false;

    B = &B[bx * BN];

    float tmp;

    #pragma unroll
    for (int k = 0; k < K; k += BK) {
        A = &A_start[k];

        // if (is_thread_0)
        // {
        //     printf("--- Sliding k=%d, B index starts %d\n", k, bx * BN + k * N);
        // }

        // cache B block
        Bs[ty * BN + tx] = B[ty * N + tx];

        // __syncthreads();
        // if (is_thread_0)
        // {
        //     for (int i_bk = 0; i_bk < BK; i_bk++)
        //     {
        //         for (int j_bn = 0; j_bn < BN; j_bn++)
        //         {
        //             printf("\tBs[%d, %d] = %f\n", i_bk, j_bn, Bs[i_bk * BN + j_bn]);
        //         }
        //     }
        //     printf("\n");
        // }

        #pragma unroll
        for (int i = 0; i < M; i += BM)
        {
            // if (is_thread_0)
            // {
            //     printf("\t--- Sliding i=%d, A index starts %d\n", i, k + i * K);
            // }

            // cache A block
            // As[tx * BK] = A[tx * K];
            As[ty * BK + tx] = A[ty * K + tx];
            __syncthreads();


            // if (is_thread_0)
            // {
            //     for (int i_bm = 0; i_bm < BM; i_bm++)
            //     {
            //         for (int j_bk = 0; j_bk < BK; j_bk++)
            //         {
            //             printf("\t\tAs[%d, %d] = %f\n", i_bm, j_bk, As[i_bm * BK + j_bk]);
            //         }
            //     }
            //     printf("\n");
            // }

            tmp = 0;
            #pragma unroll
            for (int l = 0; l < BK; l++)
            {
                tmp += As[ty * BK + l] * Bs[l * BN + tx];
            }

            

            if (k == 0) {
                // C[(i * BM + ty) * N + (bx * BN + tx)] = beta * C[(i * BM + ty) * N + (bx * BN + tx)];
                C[(i + ty) * N + (bx * BN + tx)] = beta * C[(i + ty) * N + (bx * BN + tx)];
            }

            // update correspoding C element
            // C[(i * BM + ty) * N + (bx * BN + tx)] += alpha * tmp;
            C[(i + ty) * N + (bx * BN + tx)] += alpha * tmp;


            // FMA计算需要读取缓存数据，在新一轮写入缓存前进行同步，确保所有线程计算完成
            __syncthreads();

            // if (is_thread_0)
            // {
            //     printf("\t\tcumulate tmp=%f to C[%d, %d] = %f\n", tmp, i + ty, bx * BN + tx, C[(i + ty) * N + (bx * BN + tx)]);
            // }

            // move A pointer to next block
            A += BM * K;

        }

        // move B pointer to next block
        B += BK * N;
    
    }


















    // // slide across K dimension in B
    // #pragma unroll
    // for (int k = 0; k < K; k += BK) {

        

    //     A = &A_start[k * BK];

    //     if (is_thread_0)
    //     {
    //         printf("--- Sliding k=%d, B index = %d\n", k, bx * BN + k * BK * N);
    //     }
        

    //     Bs[ty * BN + tx] = B[ty * N + tx];

    //     __syncthreads();
    //     if (is_thread_0)
    //     {
    //         for (int i = 0; i < BLOCK_SIZE; i++)
    //         {
    //             printf("\tBs[%d] = %f ", i, Bs[i]);
    //         }
    //         printf("\n");
            
    //     }


    //     // slide across M dimension in A
    //     #pragma unroll
    //     for (int i = 0; i < M; i += BM)
    //     {
    //         if (is_thread_0)
    //         {
    //             printf("\t--- Sliding i=%d, A index = %d\n", i, k * BK + i * BM * K);
    //         }
                
    //         As[tx * BK] = A[tx * K];

            

    //         // 同步所有线程缓存完成
    //         __syncthreads();

    //         if (is_thread_0)
    //         {
    //             for (int i = 0; i < BLOCK_SIZE; i++)
    //             {
    //                 printf("\t\tAs[%d] = %f ", i, As[i]);
    //             }
    //             printf("\n");
                
    //         }


            

    //         #pragma unroll
    //         for (int l = 0; l < BLOCK_SIZE; l++)
    //         {
    //             if (k % K_TILE_SIZE == 0) {
    //                 tmps[i * BM + l] = As[l] * Bs[tx];
    //             } else {
    //                 tmps[i * BM + l] += As[l] * Bs[tx];
    //             }
    //         }

    //         // #pragma unroll
    //         // for (int l = 0; l < BM; l++)
    //         // {
    //         //     tmps[k * BM + l] += block_tmp[l];
    //         // }
            

    //         // // FMA计算需要读取缓存数据，在新一轮写入缓存前进行同步，确保所有线程计算完成
    //         __syncthreads();


    //         if (is_thread_0)
    //         {
    //             for (int l = 0; l < BLOCK_SIZE; l++)
    //             {
    //                 printf("\t\ttmps[%d] (C[%d]) = %f ", l, l*N+tx, tmps[i * BM + l]);
    //             }
    //             printf("\n");
    //         }


    //         A += BM * K;

    //         // if (is_thread_0)
    //         // {
    //         //     printf("\t\tMoving A by BM * K = %d, value=%f\n", BM * K, A[0]);
    //         // }
            
            

    //     }

    //     B += BK * N;
        
    // }

    // #pragma unroll
    // for (int i = 0; i < M; i++)
    // {
    //     C[i * N + tx] = alpha * tmps[i] + beta * C[i * N + tx];
    // }











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