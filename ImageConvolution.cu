//AUTHOR : HAMORA HADI

#include<stdio.h>
#include<cuda_runtime_api.h>
#include<cuda.h>
#include<stdlib.h>
#include<assert.h>

#define Mask_width  64
#define Mask_height 64
#define Mask_radius_x Mask_width/2
#define Mask_radius_y Mask_height/2
#define TILE_WIDTH 32  
#define w_x (TILE_WIDTH + Mask_width - 1) 
#define w_y (TILE_WIDTH + Mask_height - 1)
#define clamp(x) (min(max((x), 0.0), 1.0))

float convolution_2D_OnHost(float * N,float * M,int width, int height,int i,int j);

#define wbCheck(stmt)  do {                                                    
        cudaError_t err = stmt;                                               
        if (err != cudaSuccess) {                                             
            printf( "Failed to run stmt %d ", __LINE__);                       
            printf( "Got CUDA error ...  %s ", cudaGetErrorString(err));    
            return -1;                                                        
        }                                                                     
    } while(0)

__global__ void convolution(float *I, const float* __restrict__ M, float *P,int channels, int width, int height){
   __shared__ float N_ds[w_y][w_x];
   int k;
   for (k = 0; k < channels; k++){
      int dest = threadIdx.y * TILE_WIDTH + threadIdx.x,
         destY = dest / w_x, destX = dest % w_x,
         srcY = blockIdx.y * TILE_WIDTH + destY - Mask_radius_x,
         srcX = blockIdx.x * TILE_WIDTH + destX - Mask_radius_y,
         src = (srcY * width + srcX) * channels + k;
      if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
         N_ds[destY][destX] = I[src];
      else
         N_ds[destY][destX] = 0.0;

        for (int iter=1; iter <= (w_x * w_y) / (TILE_WIDTH*TILE_WIDTH); iter++){
          // Second batch loading
          dest = threadIdx.y * TILE_WIDTH + threadIdx.x + iter*(TILE_WIDTH * TILE_WIDTH);
            destY = dest / w_x, destX = dest % w_x;
            srcY  = blockIdx.y * TILE_WIDTH + destY - Mask_radius_x;
            srcX = blockIdx.x * TILE_WIDTH + destX - Mask_radius_y;
            src = (srcY * width + srcX) * channels + k;
            if (destY < w_y && destX < w_x)
            {
                if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
                    N_ds[destY][destX] = I[src];
                else
                    N_ds[destY][destX] = 0.0;
            }
        }
      __syncthreads();

      float accum = 0;
      int y, x;
      for (y = 0; y < Mask_width; y++)
         for (x = 0; x < Mask_width; x++)
            accum += N_ds[threadIdx.y + y][threadIdx.x + x] * M[y * Mask_width + x];
      y = blockIdx.y * TILE_WIDTH + threadIdx.y;
      x = blockIdx.x * TILE_WIDTH + threadIdx.x;
      if (y < height && x < width)
         //P[(y * width + x) * channels + k] = clamp(accum);
    	  P[(y * width + x) * channels + k] = accum;
      __syncthreads();
   }
}

float convolution_2D_OnHost(float * N,float * M,int width, int height,int i,int j);

int main() {
    int maskRows=Mask_height;
    int maskColumns=Mask_width;

    int imageChannels=1;
    int imageWidth=2048;
    int imageHeight=2048;

    float * hostInputImageData;
    float * hostOutputImageData;
    float * hostMaskData;
    float * deviceInputImageData;
    float * deviceOutputImageData;
    float * deviceMaskData;
    float * outputImageOnHost;

    hostMaskData = (float *) malloc(sizeof(float)*maskRows*maskColumns);
    for(int i=0;i<maskRows*maskColumns;i++)
    	hostMaskData[i]=1.0;
    }
    //assert(maskRows == 5); 
    //assert(maskColumns == 5); 

    //To store Memory
    hostInputImageData = (float *) malloc(sizeof(float)*imageWidth*imageHeight);
    hostOutputImageData = (float *) malloc(sizeof(float)*imageWidth*imageHeight);
    outputImageOnHost = (float *) malloc(sizeof(float)*imageWidth*imageHeight);
    for(int i=0;i<imageWidth*imageHeight;i++)//To set Image data as 1.0{
    	hostInputImageData[i]=1.0;
    }

    wbCheck(cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float)));
    wbCheck(cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float)));
    wbCheck(cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float)));

    wbCheck(cudaMemcpy(deviceInputImageData,
               hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(deviceMaskData,
               hostMaskData,
               maskRows * maskColumns * sizeof(float),
               cudaMemcpyHostToDevice));

    dim3 dimGrid(((imageWidth-1)/TILE_WIDTH)+1, ((imageHeight-1)/TILE_WIDTH)+1,1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    convolution<<<dimGrid,dimBlock>>>(deviceInputImageData, deviceMaskData, deviceOutputImageData,
                                       imageChannels, imageWidth, imageHeight);

	cudaError_t err1 = cudaPeekAtLastError();
	cudaDeviceSynchronize();
	printf( "Got CUDA error ... %s \n", cudaGetErrorString(err1));

    cudaMemcpy(hostOutputImageData,
               deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost);

#if 0  //
    for(int i=0;i<imageWidth*imageHeight;i++){
        if(i>0 && (i%imageWidth==0))
            printf("\n");
    		printf("%0.2f \t",*(hostOutputImageData+i));

    		}
#endif
    for(int i=0;i<imageWidth;i++){
        			for(int j=0;j<imageHeight;j++){
        				outputImageOnHost[(i*imageWidth)+j]=convolution_2D_OnHost(hostInputImageData,hostMaskData,imageWidth,imageHeight,i,j);
        			}
        		}

#if 0  //
    for(int i=0;i<imageWidth*imageHeight;i++){
    		if(i>0 && (i%imageWidth==0))
    			printf("\n");
    		printf("%0.2f \t",*(outputImageOnHost+i));

    		}
#endif


        for(int i=0;i<imageWidth*imageHeight;i++){
        	if(outputImageOnHost[i]!=hostOutputImageData[i]){
        		 printf("\nMismatch at (Row,Col) = [%d][%d], hostComputed[]: %0.0f And device[]: %0.0f", i / imageWidth, i % imageHeight, outputImageOnHost[i], hostOutputImageData[i]);
            }
        }

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);

    free(hostMaskData);
    free(hostInputImageData);
    free(hostOutputImageData);

    printf("\n Number of Threads Per Block created in code: %d",TILE_WIDTH*TILE_WIDTH);
    printf("\n Number of Blocks Created :%d",(((imageWidth-1)/TILE_WIDTH)+1)*(((imageWidth-1)/TILE_WIDTH)+1));
    printf("No Error");
    return 0;
}

float convolution_2D_OnHost(float * N,float * M,int width, int height,int i,int j){
	float Pvalue=0.0;
	int N_start_point_i = i - (Mask_width/2);
	int N_start_point_j = j - (Mask_height/2);

	for(int k=0;k<Mask_width;k++){
		for(int l=0;l<Mask_height;l++){
			if(((N_start_point_i+k)>=0) && ((N_start_point_i+k)<width)&&((N_start_point_j+l)>=0)&&((N_start_point_j+l)<height)){
			    Pvalue+=N[(N_start_point_i+k)*width+(N_start_point_j+l)]*M[(k*Mask_width)+l];
			}
		}
	}

	return((Pvalue));
}
