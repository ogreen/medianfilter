#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <inttypes.h>

__device__ void histogramMedian(hist_type* H, const int32_t size, const int32_t medPos, im_type* retval){
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



__device__ void histogramMedianPar(hist_type* H,hist_type* Hscan,const int32_t size, const int32_t medPos, const int logSize,im_type* retval){

	int32_t tx=threadIdx.x;
	*retval=1;

	for(; tx<size;tx+=blockDim.x){
		Hscan[tx]=H[tx];
	}
	syncthreads();

	for (int32_t d=0; d<(logSize-1); d++)
	{
		tx=threadIdx.x;
		int ind=1<<(d+1);
		int activeInd=size/ind;
		for(tx=threadIdx.x; tx<activeInd;tx+=blockDim.x){
			int32_t k=tx*ind;
			Hscan[k+ind-1]=Hscan[k+ind/2-1]+Hscan[k+ind-1];
		}
		syncthreads();
	}


	Hscan[MF_HIST_SIZE-1]=0;
	syncthreads();

	for (int32_t d=logSize; d>=0; d--)
	{
		int ind=1<<(d+1);
		int activeInd=size/ind;

		for(tx=threadIdx.x; tx<activeInd;tx+=blockDim.x){
			int32_t k=tx*ind;
			hist_type temp=Hscan[k+ind/2-1];
			Hscan[k+ind/2-1]=Hscan[k+ind-1];
			Hscan[k+ind-1]=temp+Hscan[k+ind-1];
		}
		syncthreads();
	}

	tx=threadIdx.x;
	for(; tx<(size-1);tx+=blockDim.x){
		if(Hscan[tx]<medPos && Hscan[tx+1]>=medPos)
			*retval=tx;
	}
	syncthreads();
}

__device__ void histogramMedianPar256(hist_type* H,hist_type* Hscan,const int32_t size, const int32_t medPos, const int logSize,im_type* retval){

	int32_t tx=threadIdx.x;
	*retval=1;

	Hscan[tx]=H[tx];
	syncthreads();

	int32_t add;

	if(tx<128){
		add=tx<<1; Hscan[add+1]+=Hscan[add];
	}
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
	if (tx<128){
		add=tx<<1;
		temp=Hscan[add+0]; Hscan[add+0]=Hscan[add+1]; Hscan[add+1]+=temp;
	}
	syncthreads();
	tx=threadIdx.x;
	for(; tx<(size-1);tx+=blockDim.x){
		if(Hscan[tx]<medPos && Hscan[tx+1]>=medPos)
			*retval=tx;
	}
	syncthreads();
}


__device__ void histogramMedianPar128(hist_type* H,hist_type* Hscan,const int32_t size, const int32_t medPos, const int logSize,im_type* retval){

	int32_t tx=threadIdx.x;
	*retval=1;

	Hscan[tx]=H[tx];
	Hscan[tx+128]=H[tx+128];
	syncthreads();

	int32_t add;

	if(tx<128){
		add=tx<<1; Hscan[add+1]+=Hscan[add];
	}
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
	if (tx<128){
		add=tx<<1;
		temp=Hscan[add+0]; Hscan[add+0]=Hscan[add+1]; Hscan[add+1]+=temp;
	}
	syncthreads();
	tx=threadIdx.x;
	for(; tx<(size-1);tx+=blockDim.x){
		if(Hscan[tx]<medPos && Hscan[tx+1]>=medPos)
			*retval=tx;
	}
	syncthreads();
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




