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

typedef unsigned short hist_type;
typedef unsigned char im_type;

#define MF_IM_SIZE 512
#define MF_HIST_SIZE 256

#include "lookup.h"


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



__device__ void histogramAdd(hist_type* H, const hist_type * hist_col){
	int32_t tx = threadIdx.x;
	for(; tx<256;tx+=blockDim.x){
		H[tx]+=hist_col[tx];
	}
}

__device__ void histogramSub(hist_type* H, const hist_type * hist_col){
	int32_t tx = threadIdx.x;
	for(; tx<256;tx+=blockDim.x){
		H[tx]-=hist_col[tx];
	}
}

__device__ void histogramAddAndSub(hist_type* H, const hist_type * hist_colAdd,const hist_type * hist_colSub){
	int32_t tx = threadIdx.x;
	for(; tx<256;tx+=blockDim.x){
		H[tx]+=hist_colAdd[tx]-hist_colSub[tx];
	}
}

__device__ void histogramMultipleAdd(hist_type* H, const hist_type * hist_col,int histCount){
	int32_t tx = threadIdx.x;
	for(; tx<256;tx+=blockDim.x){
		hist_type temp=H[tx];;
		for(int i=0; i<histCount; i++)
		    temp+=hist_col[(i<<8)+tx];
		H[tx]=temp;
	}
}


__device__ void histogramClear(hist_type* H){
	int32_t tx = threadIdx.x;
	for(; tx<256;tx+=blockDim.x){
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




__device__ void histogramMedianPar64(hist_type* H,hist_type* Hscan,const int32_t size, const int32_t medPos, const int logSize,im_type* retval){
	int32_t tx=threadIdx.x;
	*retval=1;

	Hscan[tx]=H[tx];
	Hscan[tx+64]=H[tx+64];
	Hscan[tx+128]=H[tx+128];
	Hscan[tx+192]=H[tx+192];	
	syncthreads();

	int32_t add;

//	if(tx<128){
	add=tx<<1; Hscan[add+1]+=Hscan[add];
	add=64+tx<<1; Hscan[add+1]+=Hscan[add];

	syncthreads();
	if(tx<64){
		add=tx<<2; Hscan[add+3]+=Hscan[add+1];
	}
	syncthreads();

	if(tx<32){
		add=tx<<3;Hscan[add+7]+=Hscan[add+3];
	}
	if(tx<16){
		add=tx<<4; Hscan[add+15]+=Hscan[add+7];
	}
	if(tx<8){
		add=tx<<5; Hscan[add+31]+=Hscan[add+15];
	}
	if(tx<4){
		add=tx<<6; Hscan[add+63]+=Hscan[add+31];
	}
	if(tx<2){
		add=tx<<7; Hscan[add+127]+=Hscan[add+63];
	}
	Hscan[MF_HIST_SIZE-1]=0;
	syncthreads();

	hist_type temp;
	if (tx<1){
		add=tx<<8;
		temp=Hscan[add+127]; Hscan[add+127]=Hscan[add+255]; Hscan[add+255]+=temp;
	}
	if (tx<2){
		add=tx<<7;
		temp=Hscan[add+63]; Hscan[add+63]=Hscan[add+127]; Hscan[add+127]+=temp;
	}
	if (tx<4){
		add=tx<<6;
		temp=Hscan[add+31]; Hscan[add+31]=Hscan[add+63]; Hscan[add+63]+=temp;
	}
	if (tx<8){
		add=tx<<5;
		temp=Hscan[add+15]; Hscan[add+15]=Hscan[add+31]; Hscan[add+31]+=temp;
	}
	if (tx<16){
		add=tx<<4;
		temp=Hscan[add+7]; Hscan[add+7]=Hscan[add+15]; Hscan[add+15]+=temp;
	}
	if (tx<32){
		add=tx<<3;
		temp=Hscan[add+3]; Hscan[add+3]=Hscan[add+7]; Hscan[add+7]+=temp;
	}
	syncthreads();
	if (tx<64){
		add=tx<<2;
		temp=Hscan[add+1]; Hscan[add+1]=Hscan[add+3]; Hscan[add+3]+=temp;
	}
	syncthreads();
//	if (tx<128){
	{
		add=tx<<1;
		temp=Hscan[add+0]; Hscan[add+0]=Hscan[add+1]; Hscan[add+1]+=temp;
		
		add=64+tx<<1;
		temp=Hscan[add+0]; Hscan[add+0]=Hscan[add+1]; Hscan[add+1]+=temp;
		
	}
	syncthreads();
	tx=threadIdx.x;
	for(; tx<255;tx+=blockDim.x){
		if(Hscan[tx+1]>=medPos){
			if(Hscan[tx]<medPos){ 
				*retval=tx;
			}
			break;
			
		}
	}
	syncthreads();
}


__device__ void histogramMedianPar32Multi(hist_type* H,hist_type* Hscan,const int32_t size, const int32_t medPos, const int logSize,im_type* retval){
	int32_t tx=threadIdx.x;
	*retval=0;
	__shared__ int32_t foundIn;
	foundIn=31;
//	if(tx<32)
//	{
		Hscan[tx]=0;
		int add=tx<<3;
		for(int i=0; i<8;i++){
			Hscan[tx]+=H[add+i];
		}

		if(tx<16){
			add=tx<<1; Hscan[add+1]+=Hscan[add];
		}
		if(tx<8){
			add=tx<<2; Hscan[add+3]+=Hscan[add+1];
		}
		if(tx<4){
			add=tx<<3; Hscan[add+7]+=Hscan[add+3];
		}
		if(tx<2){
			add=tx<<4; Hscan[add+15]+=Hscan[add+7];
		}

		Hscan[31]=0;
		
		hist_type temp;
		if (tx<1){
			add=0;
			temp=Hscan[add+15]; Hscan[add+15]=Hscan[add+31]; Hscan[add+31]+=temp;
		}
		if (tx<2){
			add=tx<<4;
			temp=Hscan[add+7]; Hscan[add+7]=Hscan[add+15]; Hscan[add+15]+=temp;
		}
		if (tx<4){
			add=tx<<3;
			temp=Hscan[add+3]; Hscan[add+3]=Hscan[add+7]; Hscan[add+7]+=temp;
		}
		if (tx<8){
			add=tx<<2;
			temp=Hscan[add+1]; Hscan[add+1]=Hscan[add+3]; Hscan[add+3]+=temp;
		}
		if (tx<16){
			add=tx<<1;
			temp=Hscan[add]; Hscan[add]=Hscan[add+1]; Hscan[add+1]+=temp;
		}
	
		syncthreads();
		if(tx<31){
			if(Hscan[tx+1]>=medPos){
				if(Hscan[tx]<medPos){ 
					foundIn=tx;
				}
			}
		}	
//			if(blockIdx.x==0 && threadIdx.x==0){
//				printf("%d  %d  %d %d \n",Hscan[foundIn], Hscan[foundIn+1], foundIn,medPos);
//			}

		
		syncthreads();
		if(tx==0){
			int32_t pos=foundIn<<3;
			int32_t total=Hscan[foundIn]+H[pos];
			for(int i=0; i<8;i++)
			{
				if(total>=medPos){
					*retval=pos+i;
					break;
				}
				total+=H[pos+i+1];
			}
//			if(blockIdx.x==0){
//				for(int i=0; i<32; i++)
//					printf("%d ", Hscan[i]);
//				printf("      -  %d  %d  %d  %d\n",foundIn, foundIn<<3,Hscan[foundIn], total);
//			}
			
		}
//	}



	/*
	syncthreads();
	tx=threadIdx.x;
	for(; tx<255;tx+=blockDim.x){
		if(Hscan[tx+1]>=medPos){
			if(Hscan[tx]<medPos){ 
				*retval=tx;
			}
			break;
			
		}
	}
	syncthreads();
*/	
}

__device__ void histogramMedianPar32WorkInefficient(hist_type* H,hist_type* Hscan,const int32_t size, const int32_t medPos, const int logSize,im_type* retval){
	int32_t tx=threadIdx.x;
	*retval=0;
	__shared__ int32_t foundIn;
	foundIn=31;
        if (tx>32)
	  return;
	
	//if(tx<32)
	{
		  Hscan[tx]=0;
		int add=tx<<3;
		for(int i=0; i<8;i++){
	        	Hscan[tx]+=H[add+i];
		}
		syncthreads();
		if(tx>1){
		  Hscan[tx]+=Hscan[tx-1];
		}
 		if(tx>=2){
		  Hscan[tx]+=Hscan[tx-2];
		}
    		if(tx>=4){
		  Hscan[tx]+=Hscan[tx-4];
		}
 		if(tx>=8){
		  Hscan[tx]+=Hscan[tx-8];
		}
 		if(tx>=16){
		  Hscan[tx]+=Hscan[tx-16];
		}
 
	}
/*	       		if(blockIdx.x==0 && threadIdx.x==0){
				for(int i=0; i<32; i++)
					printf("%d ", Hscan[i]);
		//		printf("      -  %d \n",Hscan[i]);
		        	printf("\n");
			}
*/       
		syncthreads();
		if(tx<31){
			if(Hscan[tx+1]>=medPos){
				if(Hscan[tx]<medPos){ 
					foundIn=tx;
				}
			}
		}	
		syncthreads();
/*       	                         			if(blockIdx.x==0 && threadIdx.x==0){
				printf("%d  %d  %d %d \n",Hscan[foundIn], Hscan[foundIn+1], foundIn,medPos);
			      
				int start=max(0,(foundIn-1)<<3);
				int stop=min(256, (foundIn+2)<<3);
				printf("%d %d\n", start,stop);
				for(int i=start; i<stop;i++)
				  printf ("%d ",H[i]);
			      printf("\n"); 			
			      }
*/
		syncthreads();
		if(tx==0){
			int32_t total=Hscan[foundIn];
			int32_t pos=(foundIn+1)<<3;
		        total+=H[pos];     
			
/*			if(blockIdx.x==0 && threadIdx.x==0){
				printf(" %d \n",total );
			} 
*/			*retval=pos+8;
			for(int i=0; i<8;i++)
			{
				if(total>=medPos){
					*retval=pos+i;
					break;
				}
				total+=H[pos+i];
/* 			if(blockIdx.x==0 && threadIdx.x==0){
				printf(" %d \n",total );
			}                            
*/        		}
       		
		}
//	}

     
}
 

/////////////////////////////////////////
/////////////////////////////////////////
/////////////////////////////////////////

__global__ void cuMedianFilter (im_type* src, im_type* dest, hist_type * hist, int32_t rows, int32_t cols, int32_t r, int32_t medPos)
{
    __shared__ hist_type H[MF_HIST_SIZE];
    __shared__ hist_type Hscan[MF_HIST_SIZE];
    __shared__ im_type retval;


    for (int32_t j=threadIdx.x; j<cols; j+=blockDim.x){
    	hist[j*MF_HIST_SIZE+src[j]]=r+2;
    }
    syncthreads();

    for(int i=1; i< r; i++){
    	int32_t pos=min(i,rows-1);
        for (int32_t j=threadIdx.x; j<cols; j+=blockDim.x){
        	hist[j*MF_HIST_SIZE+src[pos*cols+j]]++;
        }
        syncthreads();
    }


    for(int i=0; i< rows; i++){
        histogramClear(H); syncthreads();

        int32_t possub=max(0,i-r-1);
        int32_t posadd=min(rows-1,i+r);

        for (int32_t j=threadIdx.x; j<cols; j+=blockDim.x){
        	hist[MF_HIST_SIZE*j+ src[possub*cols+j] ]--;
        	hist[MF_HIST_SIZE*j+ src[posadd*cols+j] ]++;
        }
        syncthreads();

        for(int32_t j=0;j<(2*r);j++){
        	histogramAdd(H,hist+MF_HIST_SIZE*j); syncthreads();
        }

        for(int32_t j=r;j<cols-r;j++){
            int32_t possub=max(j-r,0);
            int32_t posadd=min(j+r,cols-1);
            histogramAdd(H, hist+posadd*MF_HIST_SIZE); syncthreads();
//            histogramMedian(H,MF_HIST_SIZE,medPos, &retval);
//            histogramMedianPar(H,Hscan,MF_HIST_SIZE,medPos, 8,&retval);
//            histogramMedianPar256(H,Hscan,MF_HIST_SIZE,medPos, 8,&retval);

            histogramMedianPar32WorkInefficient(H,Hscan,MF_HIST_SIZE,medPos, 8,&retval);
//            return;
            syncthreads();

            if(threadIdx.x==0){
            	dest[i*cols+j]=retval;
           }
                syncthreads();

            histogramSub(H, hist+possub*MF_HIST_SIZE); syncthreads();
        }
        syncthreads();
    }
}


__global__ void cuMedianFilterMultiBlock (im_type* src, im_type* dest, hist_type * histPar, int32_t rows, int32_t cols, int32_t r, int32_t medPos)
{
    __shared__ hist_type H[MF_HIST_SIZE];
    __shared__ hist_type Hscan[32];
    __shared__ im_type retval;

    int32_t extraRowThread=rows%gridDim.x;
    int32_t doExtraRow=blockIdx.x<extraRowThread;
    int32_t startRow=0, stopRow=0;
    int32_t rowsPerBlock= rows/gridDim.x+doExtraRow;

    // The following code partitions the work to the blocks. Some blocks will do one row more
	// than other blocks. This code is responsible for doing that balancing
	if(doExtraRow){
        startRow=rowsPerBlock*blockIdx.x;
        stopRow=min(rows, startRow+rowsPerBlock);
    }
    else{
        startRow=(rowsPerBlock+1)*extraRowThread+(rowsPerBlock)*(blockIdx.x-extraRowThread);    
        stopRow=min(rows, startRow+rowsPerBlock);        
    }

    hist_type* hist=histPar+cols*MF_HIST_SIZE*blockIdx.x;
    if (blockIdx.x==(gridDim.x-1))
    	stopRow=rows;
    syncthreads();
    int32_t initNeeded=0, initVal, initStartRow, initStopRow;

    if(blockIdx.x==0){
    	initNeeded=1; initVal=r+2; initStartRow=1;	initStopRow=r;
    }
    else if (startRow<(r+2)){
    	//initNeeded=1; initVal=r+2-startRow-1; initStartRow=1+startRow;	initStopRow=r+startRow+1;
    	initNeeded=1; initVal=r+2-startRow; initStartRow=1;	initStopRow=r+stopRow-startRow;
    }
    else{
    	initNeeded=0; initVal=0; initStartRow=startRow-(r+1);	initStopRow=r+startRow;    	
    }
   syncthreads();
//   int counter=0;
    // In the original algorithm an initialization phase was required as part of the window was outside the
	// image. In this parallel version, the initializtion is required for all thread blocks that part
	// of the median filter is outside the window.
	// For all threads in the block the same code will be executed.
	if (initNeeded){
		for (int32_t j=threadIdx.x; j<cols; j+=blockDim.x){
			hist[j*MF_HIST_SIZE+src[j]]=initVal;
		}
//		counter+=initVal;
    }
    syncthreads();
    
	// Fot all remaining rows in the median filter, add the values to the the histogram
	for (int32_t j=threadIdx.x; j<cols; j+=blockDim.x){
		for(int i=initStartRow; i<initStopRow; i++){
			int32_t pos=min(i,rows-1);
				hist[j*MF_HIST_SIZE+src[pos*cols+j]]++;
			}
	}
    
  syncthreads();
//     if(threadIdx.x==0 && initNeeded)
//      printf("%d, %d, %d, %d \n",blockIdx.x, startRow,stopRow, counter);


	 // Going through all the rows that the block is responsible for.
	 int32_t inc=blockDim.x*MF_HIST_SIZE;
     for(int i=startRow; i< stopRow; i++){
         // For every new row that is started the global histogram for the entire window is restarted.
		 histogramClear(H);
		 // Computing some necessary indices
         int32_t possub=max(0,i-r-1),posadd=min(rows-1,i+r);
		 int32_t possubMcols=possub*cols, posaddMcols=posadd*cols;
         syncthreads();
         int32_t histPos=threadIdx.x*MF_HIST_SIZE;
		 // Going through all the elements of a specific row. Foeach histogram, a value is taken out and 
		 // one value is added.
         for (int32_t j=threadIdx.x; j<cols; j+=blockDim.x){
          	hist[histPos+ src[possubMcols+j] ]--;
          	hist[histPos+ src[posaddMcols+j] ]++;
          	histPos+=inc;
			syncthreads();
         }

         
         histogramMultipleAdd(H,hist, 2*r+1);         

         syncthreads();         	
         int32_t rowpos=i*cols;
         int32_t cols_m_1=cols-1;
         for(int32_t j=r;j<cols-r;j++){
             int32_t possub=max(j-r,0);
             int32_t posadd=min(j+1+r,cols_m_1);
             //histogramMedianPar32Multi(H,Hscan,MF_HIST_SIZE,medPos, 8,&retval);
             histogramMedianPar32WorkInefficient(H,Hscan,MF_HIST_SIZE,medPos, 8,&retval);
             syncthreads();
             if (threadIdx.x==0)
            	 dest[rowpos+j]=retval;

             //histogramAddAndSub(H, hist+posadd*MF_HIST_SIZE,hist+possub*MF_HIST_SIZE);                 
             histogramAddAndSub(H, hist+(int)(posadd<<8),hist+(int)(possub<<8));                 
             syncthreads();
        }
         syncthreads();
     }

}


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

	cudaSetDevice(1);

	// Allocating host and device memory.
	hostSrc=(im_type*)malloc(memBytesImage);
	hostDest=(im_type*)malloc(memBytesImage);
	hostRef=(im_type*)malloc(memBytesImage);
	CUDA(cudaMalloc((void**)(&devSrc), memBytesImage));
	CUDA(cudaMalloc((void**)(&devDest), memBytesImage));
	CUDA(cudaMalloc((void**)(&devHist), memBytesHist));

	// Loading image from file
	char filename[]="barbara.csv";
	readImage(filename, hostSrc, rows, cols);

	// Copying data from host to device.
	CUDA(cudaMemcpy(devSrc,hostSrc,memBytesImage,cudaMemcpyHostToDevice));
//	CUDA(cudaMemcpy(devDest,hostSrc,memBytesImage,cudaMemcpyHostToDevice));
	CUDA(cudaMemset(devDest,0,memBytesImage));
	CUDA(cudaMemset(devHist,0,memBytesHist));


	int32_t kernel=5;
	int32_t r=(kernel-1)/2;
	int32_t medPos=2*r*r+2*r;

	// Setting CUDA kernel properties.
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


	// Copying filtered image back from the device to the host.
	CUDA(cudaMemcpy(hostDest,devDest,memBytesImage,cudaMemcpyDeviceToHost));

	// Loading reference image for comparison purposes.
	char refFileName[]="ref.csv";
	readImage(refFileName, hostRef, rows, cols);
	float psnrval = psnr(hostDest, hostRef, rows, cols);
	printf("Time: %f  (secs) \t\t PSNR: %f \n",time/1000,psnrval);


	for(int gridSize=32; gridSize<512; gridSize+=32)
	//for(int gridSize=32; gridSize<64; gridSize+=32)
	{
		// Setting CUDA kernel properties.
		dim3 gridDim; gridDim.x=gridSize;
		dim3 blockDim; blockDim.x=64;
		hist_type* devHistMulti;
		int32_t memBytesHistMulti=sizeof(hist_type)*cols*MF_HIST_SIZE*gridDim.x;
		CUDA(cudaMalloc((void**)(&devHistMulti), memBytesHistMulti));
		CUDA(cudaMemset(devDest,0,memBytesImage));
		CUDA(cudaMemset(devHistMulti,0,memBytesHistMulti));

		cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
		
		cudaEvent_t start,stop; float multiTime;
		cudaEventCreate(&start); cudaEventCreate(&stop);
		cudaEventRecord(start,0);
		cuMedianFilterMultiBlock<<<gridDim,blockDim>>>(devSrc, devDest, devHistMulti, rows, cols, r, medPos);
		cudaEventRecord(stop,0);
		cudaEventSynchronize(stop);
		cudaThreadSynchronize();
		cudaEventElapsedTime(&multiTime, start, stop);

		cudaFree(devHistMulti);

		// Copying filtered image back from the device to the host.
		CUDA(cudaMemcpy(hostDest,devDest,memBytesImage,cudaMemcpyDeviceToHost));

		float psnrval = psnr(hostDest, hostRef, rows, cols);
		printf("Time: %f  (secs) \t\t PSNR: %f \t\t Speedup %f\n",multiTime/1000,psnrval,time/multiTime);

	}

	// Deallocating host and device memory.
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
	int32_t temp;
	for(int32_t r=0; r<rows;r++){
		for(int32_t c=0; c<cols; c++){
			int read=fscanf(file, "%d ", &temp);
			im_type stam=temp;
			imread[r*cols+c]=(im_type)stam;
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
//			printf("%d ",(f1-f2)*(f1-f2));
		}
//		printf("\n");
	}
	if (mse==0)
		return 100;
//	printf("%ld ", mse);
	float fmse=mse/(cols*rows);
	return 20*log10(255/sqrt(fmse) );

}
