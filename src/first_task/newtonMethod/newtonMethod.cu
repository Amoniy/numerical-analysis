#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cuda_runtime.h>

#define BOUND     2.0
#define EPS       1e-9

#define ITERS     16384
#define MAS_DIM   8192
#define BLOCK_DIM 16

#define complexEq(rel, iml, rer, imr) \
    abs(rel - rer) < EPS && abs(iml - imr) < EPS

__global__ void
getRoot(unsigned char * o_data)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    double cur_re = (double)(x - MAS_DIM / 2) / (MAS_DIM / (BOUND * 2));
    double cur_im = (double)(y - MAS_DIM / 2) / (MAS_DIM / (BOUND * 2));

    double new_re;
    double new_im;

    double num_re;
    double num_im;
    
    double div_re;
    double div_im;

    double div;

    for (unsigned int i = 0; i < ITERS; ++i)
    {
        // new_z = cur_z - (cur_z^3 - 1) / (3 * cur_z^2)
        div = 3.0 * (cur_re * cur_re * cur_re * cur_re +
                    2.0 * cur_re * cur_re * cur_im * cur_im +
                    cur_im * cur_im * cur_im * cur_im);
        
        if (complexEq(div, 0.0, 0.0, 0.0))
        {
            o_data[y * MAS_DIM + x] = 0 + '0';
            return;
        }

        div_re = cur_re * cur_re - cur_im * cur_im;
        div_im = 2.0 * cur_re * cur_im;

        num_re = cur_re * cur_re * cur_re - 3.0 * cur_re * cur_im * cur_im - 1.0;
        num_im = 3.0 * cur_re * cur_re * cur_im - cur_im * cur_im * cur_im;
        
        new_re = cur_re - (num_re * div_re + num_im * div_im) / div;
        new_im = cur_im - (div_re * num_im - div_im * num_re) / div;

        if (complexEq(new_re, new_im, cur_re, cur_im))
        {
            if (complexEq(new_re, new_im, 1.0, 0.0))
            {
                o_data[y * MAS_DIM + x] = 1 + '0';
            }
            else if (complexEq(new_re, new_im, -0.5, sqrt(3.0) / 2.0))
            {
                o_data[y * MAS_DIM + x] = 2 + '0';
            }
            else if (complexEq(new_re, new_im, -0.5, -sqrt(3.0) / 2.0))
            {
                o_data[y * MAS_DIM + x] = 3 + '0';
            }
            else
            {
                o_data[y * MAS_DIM + x] = 0 + '0';
            }
            return;
        }

        cur_re = new_re;
        cur_im = new_im;
    }

    o_data[y * MAS_DIM + x] = 0 + '0';
}

int
main(int argc, char **argv)
{
    cudaError_t err = cudaSuccess;

    dim3 dimsOut(MAS_DIM, MAS_DIM);
    unsigned int mem_size_Out = dimsOut.x * dimsOut.y * sizeof(unsigned char);
    unsigned char * h_Out = (unsigned char *)malloc(mem_size_Out);
    
    if (h_Out == NULL)
    {
        fprintf(stderr, "Failed to allocate host matrix!\n");
        exit(EXIT_FAILURE);
    }
    
    unsigned char * d_Out = NULL;
    err = cudaMalloc((void **)&d_Out, mem_size_Out);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device matrix (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    dim3 threads(BLOCK_DIM, BLOCK_DIM);
    dim3 grid(dimsOut.x / threads.x, dimsOut.y / threads.y);

    printf("Computing result using CUDA Kernel...\n");

    getRoot<<<grid, threads>>>(d_Out);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch getRoot kernel (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Copying output data from the CUDA device to the host memory...\n");
    err = cudaMemcpy(h_Out, d_Out, mem_size_Out, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy matrix from device to host (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_Out);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device matrix (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Writing output data on disk...\n");
    FILE * output;
    output = fopen("point_mas.txt", "w");
    for (size_t i = 0; i < MAS_DIM; ++i)
    {
        fwrite(h_Out + i * MAS_DIM, MAS_DIM, 1, output);
        fwrite("\n", 1, 1, output);
    }
    fclose(output);

    free(h_Out);

    printf("Done!\n");
    return 0;
}
