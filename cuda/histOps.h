
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
		hist_type temp=H[tx];
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


__device__ void histogramAdd16(hist_type* H, const hist_type * hist_col){
	int32_t tx = threadIdx.x;
	if (tx<16){
		H[tx]+=hist_col[tx];
	}
}
__device__ void histogramSub16(hist_type* H, const hist_type * hist_col){
	int32_t tx = threadIdx.x;
	if (tx<16){
		H[tx]-=hist_col[tx];
	}
}
__device__ void histogramAddAndSub16(hist_type* H, const hist_type * hist_colAdd,const hist_type * hist_colSub){
	int32_t tx = threadIdx.x;
	if (tx<16){
		H[tx]+=hist_colAdd[tx]-hist_colSub[tx];
	}
}
__device__ void histogramMultipleAdd16(hist_type* H, const hist_type * hist_col,int histCount){
	int32_t tx = threadIdx.x;
	if (tx<16){
		hist_type temp=H[tx];
		for(int i=0; i<histCount; i++)
		    temp+=hist_col[(i<<4)+tx];
		H[tx]=temp;
	}
}

__device__ void histogramClear16(hist_type* H){
	int32_t tx = threadIdx.x;
	if (tx<16){
		H[tx]=0;
	}
}



__device__ void histogramAdd8(hist_type* H, const hist_type * hist_col){
	int32_t tx = threadIdx.x;
	if (tx<8){
		H[tx]+=hist_col[tx];
	}
}
__device__ void histogramSub8(hist_type* H, const hist_type * hist_col){
	int32_t tx = threadIdx.x;
	if (tx<8){
		H[tx]-=hist_col[tx];
	}
}
__device__ void histogramAddAndSub8(hist_type* H, const hist_type * hist_colAdd,const hist_type * hist_colSub){
	int32_t tx = threadIdx.x;
	if (tx<8){
		H[tx]+=hist_colAdd[tx]-hist_colSub[tx];
	}
}
__device__ void histogramMultipleAdd8(hist_type* H, const hist_type * hist_col,int histCount){
	int32_t tx = threadIdx.x;
	if (tx<8){
		hist_type temp=H[tx];
		for(int i=0; i<histCount; i++)
		    temp+=hist_col[(i<<3)+tx];
		H[tx]=temp;
	}
}

__device__ void histogramClear8(hist_type* H){
	int32_t tx = threadIdx.x;
	if (tx<8){
		H[tx]=0;
	}
}


__device__ void histogramAdd32(hist_type* H, const hist_type * hist_col){
	int32_t tx = threadIdx.x;
	if (tx<32){
		H[tx]+=hist_col[tx];
	}
}
__device__ void histogramSub32(hist_type* H, const hist_type * hist_col){
	int32_t tx = threadIdx.x;
	if (tx<32){
		H[tx]-=hist_col[tx];
	}
}
__device__ void histogramAddAndSub32(hist_type* H, const hist_type * hist_colAdd,const hist_type * hist_colSub){
	int32_t tx = threadIdx.x;
	if (tx<32){
		H[tx]+=hist_colAdd[tx]-hist_colSub[tx];
	}
}
__device__ void histogramMultipleAdd32(hist_type* H, const hist_type * hist_col,int histCount){
	int32_t tx = threadIdx.x;
	if (tx<32){
		hist_type temp=H[tx];
		for(int i=0; i<histCount; i++)
		    temp+=hist_col[(i<<5)+tx];
		H[tx]=temp;
	}
}

__device__ void histogramClear32(hist_type* H){
	int32_t tx = threadIdx.x;
	if (tx<32){
		H[tx]=0;
	}
}