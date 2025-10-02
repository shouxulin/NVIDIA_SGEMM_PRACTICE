#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <utils.cuh>
#include <cstring>
#include <iostream>
#include <fstream>

#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

int main(int argc, char **argv) {

    int kernel_num = 8;
    int dim = 256;
    int warmup_times = 5;
    int repeat_times = 10;
    int is_print_matrix = 0;
    int verify = 0;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--kernel" && i + 1 < argc) {
            kernel_num = atoi(argv[++i]);
        } else if (arg == "--dim" && i + 1 < argc) {
            dim = atoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            warmup_times = atoi(argv[++i]);
        } else if (arg == "--repeat" && i + 1 < argc) {
            repeat_times = atoi(argv[++i]);
        } else if (arg == "--print" && i + 1 < argc) {
            is_print_matrix = atoi(argv[++i]);
        } else if (arg == "--verify" && i + 1 < argc) {
            verify = atoi(argv[++i]);
        } else {
            printf("Unknown or incomplete argument: %s\n", arg.c_str());
            exit(EXIT_FAILURE);
        }
    }

    std::cout << "kernel_num: " << kernel_num << std::endl;
    std::cout << "dim: " << dim << std::endl;
    std::cout << "warmup_times: " << warmup_times << std::endl;
    std::cout << "repeat_times: " << repeat_times << std::endl;
    std::cout << "is_print_matrix: " << is_print_matrix << std::endl;
    std::cout << "verify: " << verify << std::endl;


    // 申明句柄，创建句柄, cublasCreate会返回一个cublasStatus_t类型的值，用来判断句柄是否创建成功(值为0)
    cublasHandle_t handle;
    if (cublasCreate(&handle)) {
        printf("Create cublas handle error.\n");
        exit(EXIT_FAILURE);
    };

    // 采用cudaEvent进行gpu流计时，cudaEvent相当于在目标流中发布事件任务
    float elapsed_time;
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);

    

    int m, n, k;

    float alpha = 1.0, beta = 0.; //two arbitary input parameters，C=α*AB+β*C

    float *A[warmup_times+repeat_times] = {NULL}, *B[warmup_times+repeat_times] = {NULL}, *C[warmup_times+repeat_times] = {NULL}, *C_ref[warmup_times+repeat_times] = {NULL};     //host matrices
    float *dA[warmup_times+repeat_times] = {NULL}, *dC[warmup_times+repeat_times] = {NULL}, *dC_ref[warmup_times+repeat_times] = {NULL}; //device matrices

    for (int i = 0; i < warmup_times + repeat_times; i++)
    {
        A[i] = (float *) malloc(sizeof(float) * dim * dim);
        B[i] = (float *) malloc(sizeof(float) * dim * dim);
        C[i] = (float *) malloc(sizeof(float) * dim * dim);
        C_ref[i] = (float *) malloc(sizeof(float) * dim * dim);

        randomize_matrix(A[i], dim * dim);
        randomize_matrix(B[i], dim * dim);
        randomize_matrix(C[i], dim * dim);
        copy_matrix(C[i], C_ref[i], dim * dim);

        cudaCheck(cudaMalloc((void **) &dA[i], sizeof(float) * dim * dim));
        cudaCheck(cudaMalloc((void **) &dC[i], sizeof(float) * dim * dim));
        cudaCheck(cudaMalloc((void **) &dC_ref[i], sizeof(float) * dim * dim));


        cudaCheck(cudaMemcpy(dA[i], A[i], sizeof(float) * dim * dim, cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(dC[i], C[i], sizeof(float) * dim * dim, cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(dC_ref[i], C_ref[i], sizeof(float) * dim * dim, cudaMemcpyHostToDevice));
    }
    

    m = n = k = dim;
    printf("m=n=k=%d\n", m);

    // warm up
    cudaDeviceSynchronize();
    for (int i = 0; i < warmup_times; i++)
    {
        test_kernel(kernel_num, m, n, k, alpha, dA[i], B[i], beta, dC[i], handle);
        if (verify) {
            test_kernel(0, m, n, k, alpha, dA[i], B[i], beta, dC_ref[i], handle);      // cuBLAS
            cudaDeviceSynchronize();
            cudaMemcpy(C[i], dC[i], sizeof(float) * m * n, cudaMemcpyDeviceToHost);
            cudaMemcpy(C_ref[i], dC_ref[i], sizeof(float) * m * n, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            
            if (is_print_matrix){
                printf("C from mysgemm:\n");
                print_matrix(C[i], m, n);
                printf("\n");
                printf("C from cuBLAS:\n");
                print_matrix(C_ref[i], m, n);
            }

            if (!verify_matrix(C_ref[i], C[i], m * n)) {
                printf("Failed to pass the correctness verification against NVIDIA cuBLAS. Exited.\n");
                exit(EXIT_FAILURE);
            } else{
                printf("Passed the correctness verification against NVIDIA cuBLAS.\n");
            }

        }
    }

    cudaDeviceSynchronize();

    cudaEventRecord(beg);
    for (int j = 0; j < repeat_times; j++) {
        test_kernel(kernel_num, m, n, k, alpha, dA[j + warmup_times], B[j + warmup_times], beta, dC[j + warmup_times], handle);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, beg, end);
    elapsed_time /= 1000.; //换算成秒

    printf("Average elasped time: (%f) second, performance: (%f) GFLOPS. size: (%d).\n",
            elapsed_time / repeat_times, 2. * 1e-9 * repeat_times * m * n * k / elapsed_time, m);
    fflush(stdout);
    
    
    

    // 释放CPU和GPU空间
    for (int i = 0; i < warmup_times + repeat_times; i++)
    {
        free(A[i]);
        free(B[i]);
        free(C[i]);
        free(C_ref[i]);
        cudaFree(dA[i]);
        cudaFree(dC[i]);
        cudaFree(dC_ref[i]);
    }
    return 0;
};
