/*
 ============================================================================
 Name        : cuMedianFilter.cu
 Author      : Oded Green
 Version     :
 Copyright   : BSD 3-Clause
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <inttypes.h>

#define TEST
// THREADS is defined to be the max threads
#define MAX_THREADS 256
#define SINGLE_BLOCK 0
#define PRINT_ON 0

int print_u, print_v;

typedef int32_t hist_type;
typedef int32_t im_type;

#define MF_IM_SIZE 512
#define MF_HIST_SIZE 256


#define CUDA(call, ...) do {                        \
        cudaError_t _e = (call);                    \
        if (_e == cudaSuccess) break;               \
        fprintf(stdout,                             \
                "CUDA runtime error: %s (%d)\n",    \
                cudaGetErrorString(_e), _e);        \
        return -1;                                  \
    } while (0)

#define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })
#define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _b : _a; })



__device__ void histogramAdd(hist_type* H, const hist_type * hist_col, const int size){
	int32_t tx = threadIdx.x;
	for(; tx<size;tx+=blockDim.x){
		H[tx]+=hist_col[tx];
	}
}
__device__ void histogramSub(hist_type* H, const hist_type * hist_col, const int size){
	int32_t tx = threadIdx.x;
	for(; tx<size;tx+=blockDim.x){
		H[tx]-=hist_col[tx];
	}
}

__device__ void histogramClear(hist_type* H, const int size){
	int32_t tx = threadIdx.x;
	for(; tx<size;tx+=blockDim.x){
		H[tx]=0;
	}
}

__device__ void histogramClearAllColmuns(hist_type* hist, const int32_t columns,const int32_t hist_size){
	int32_t tx = threadIdx.x;
	int array_size=columns*hist_size;
	for(; tx<array_size;tx+=blockDim.x){
		hist[tx]=0;
	}
}


__device__ void historgramMedian(hist_type* H, const int32_t size, const int32_t medPos, int32_t* retval){
	int32_t tx=threadIdx.x;
	if(tx==0){
		int32_t sum=0;
		*retval=size;
		for(int32_t i=0; i<size; i++){
			sum+=H[i];
			if(sum>=medPos){
				*retval=i;
				return;
			}
		}

	}
}

__device__ void ClearDest(im_type* dest,int32_t rows,int32_t cols){
	int32_t tx=threadIdx.x;
	int32_t totalPixels=rows*cols;
	for(; tx<totalPixels;tx+=blockDim.x)
		dest[tx]=0;
}

/////////////////////////////////////////
/////////////////////////////////////////
/////////////////////////////////////////

__global__ void cuMedianFilter (im_type* src, im_type* dest, hist_type * hist, int32_t rows, int32_t cols, int32_t r, int32_t medPos)
{
    __shared__ int32_t H[MF_HIST_SIZE];
    histogramClearAllColmuns(hist,cols,MF_HIST_SIZE);
    ClearDest(dest,rows,cols);
    syncthreads();

    for (int32_t j=threadIdx.x; j<cols; j+=blockDim.x){
    	hist[j*MF_HIST_SIZE+src[j]]=r+2;
    }
    syncthreads();

    for(int i=0; i< r; i++){
    	int32_t pos=min(i,rows-1);
        for (int32_t j=threadIdx.x; j<cols; j+=blockDim.x){
//        	atomicAdd(hist+j*MF_HIST_SIZE+src[pos*cols+j],1);
        	hist[j*MF_HIST_SIZE+src[pos*cols+j]]++;
        }
        syncthreads();

    }

    for(int i=0; i< rows; i++){
        histogramClear(H, MF_HIST_SIZE);

        int32_t possub=max(0,i-r-1);
        int32_t posadd=min(rows-1,i+r);
        syncthreads();

        for (int32_t j=threadIdx.x; j<cols; j+=blockDim.x){
//        	atomicSub(hist+MF_HIST_SIZE*j+ src[possub*cols+j],1);
//        	atomicAdd(hist+MF_HIST_SIZE*j+ src[posadd*cols+j],1);

        	hist[MF_HIST_SIZE*j+ src[possub*cols+j] ]--;
        	hist[MF_HIST_SIZE*j+ src[posadd*cols+j] ]++;
        }
        syncthreads();

        for(int32_t j=0;j<(2*r);j++){
        	histogramAdd(H,hist+MF_HIST_SIZE*j,MF_HIST_SIZE);
            syncthreads();
        }

        for(int32_t j=r;j<cols-r;j++){
            int32_t possub=max(j-r,0);
            int32_t posadd=min(j+r,cols-1);
            histogramAdd(H, hist+posadd*MF_HIST_SIZE, MF_HIST_SIZE);
            syncthreads();
            int32_t retval;
            historgramMedian(H,MF_HIST_SIZE,medPos, &retval);
            syncthreads();

            if(threadIdx.x==0){
           	dest[i*cols+j]=retval;
            }

            histogramSub(H, hist+possub*MF_HIST_SIZE, MF_HIST_SIZE);
            syncthreads();

        }
        syncthreads();

    }

}


//int array

void readImage(char* filename,im_type* imread, int32_t rows, int32_t cols);
float psnr(im_type* im1,im_type* im2, int32_t rows, int32_t cols);


int main(const int argc, char *argv[])
{
	im_type *hostSrc=NULL,*hostDest=NULL, *hostRef;
	im_type *devSrc=NULL,*devDest=NULL;

	int32_t rows=MF_IM_SIZE,cols=MF_IM_SIZE;
	int32_t pixels=rows*cols;
	int32_t memBytesImage=sizeof(im_type)*pixels;

	hist_type* devHist;
	int32_t memBytesHist=sizeof(hist_type)*cols*MF_HIST_SIZE;

	hostSrc=(im_type*)malloc(memBytesImage);
	hostDest=(im_type*)malloc(memBytesImage);
	hostRef=(im_type*)malloc(memBytesImage);

	readImage("barbara.csv", hostSrc, rows, cols);


	CUDA(cudaMalloc((void**)(&devSrc), memBytesImage));
	CUDA(cudaMalloc((void**)(&devDest), memBytesImage));
	CUDA(cudaMalloc((void**)(&devHist), memBytesHist));

	CUDA(cudaMemcpy(devSrc,hostSrc,memBytesImage,cudaMemcpyHostToDevice));


	int32_t kernel=5;
	int32_t r=(kernel-1)/2;
	int32_t medPos=2*r*r+2*r;

	dim3 gridDim; gridDim.x=1;
	dim3 blockDim; blockDim.x=256;
	cudaEvent_t start,stop; float time;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	cudaEventRecord(start,0);
	cuMedianFilter<<<gridDim,blockDim>>>(devSrc, devDest, devHist, rows, cols, r, medPos);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaThreadSynchronize();
	cudaEventElapsedTime(&time, start, stop);

	printf("Time: %f  (secs)\n",time/1000);
	CUDA(cudaMemcpy(hostDest,devDest,memBytesImage,cudaMemcpyDeviceToHost));

	readImage("ref.csv", hostRef, rows, cols);
	float psnrval = psnr(hostDest, hostRef, rows, cols);
	printf("\npsnr: %f\n", psnrval);

	cudaFree(devHist);
	cudaFree(devSrc);
	cudaFree(devDest);
	free(hostRef);
	free(hostSrc);
	free(hostDest);

    return 0;
}

void readImage(char* filename,im_type* imread, int32_t rows, int32_t cols)
{
	FILE* file=fopen(filename,"r");
	im_type temp;
	for(int32_t r=0; r<rows;r++){
		for(int32_t c=0; c<cols; c++){
			fscanf(file, "%d ", &temp);
			imread[r*cols+c]=temp;
		}
	}

	fclose(file);
}

float psnr(im_type* im1,im_type* im2, int32_t rows, int32_t cols){
	int64_t mse=0;
	for(int32_t r=0; r<rows;r++){
		for(int32_t c=0; c<cols; c++){
			int32_t f1=im1[r*cols+c];
			int32_t f2=im2[r*cols+c];
			mse+= (f1-f2)*(f1-f2);
		}
	}
	if (mse==0)
		return 100;
	float fmse=mse/(cols*rows);
	return 20*log10(255/sqrt(fmse) );

}
