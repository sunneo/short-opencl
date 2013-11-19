#include "opencl_runtime.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

int main(){
   size_t dimglobal[3]={2048,1,1};
   size_t dimlocal[3]={256,1,1};
   int i,ierr=0,iNumElements = 16384; 

   float *d_a,*d_b,*d_c, *h_a, *h_b, *h_c;
   h_a = (float*)malloc(sizeof(float)*iNumElements);
   h_b = (float*)malloc(sizeof(float)*iNumElements);
   h_c = (float*)malloc(sizeof(float)*iNumElements);
   openclInitFromFile("VectorAdd.cl");
   openclMalloc((void**)&d_a,sizeof(float)*iNumElements);
   openclMalloc((void**)&d_b,sizeof(float)*iNumElements);
   openclMalloc((void**)&d_c,sizeof(float)*iNumElements);
   for(i=0; i<iNumElements; ++i){
      h_a[ i ] = 1.0;  
      h_b[ i ] = 2.0;
   }
   openclMemcpy(d_a,h_a,sizeof(float)*iNumElements,openclMemcpyHostToDevice);
   openclMemcpy(d_b,h_b,sizeof(float)*iNumElements,openclMemcpyHostToDevice);
   openclLaunchKernel("VectorAdd",dimglobal,dimlocal,d_a,d_b,d_c,iNumElements);
   openclThreadSynchronize();
   openclMemcpy(h_a,d_a, sizeof(float)*iNumElements, openclMemcpyDeviceToHost);
   openclMemcpy(h_b,d_b, sizeof(float)*iNumElements, openclMemcpyDeviceToHost);
   openclMemcpy(h_c,d_c, sizeof(float)*iNumElements, openclMemcpyDeviceToHost);
   openclFree(d_a);
   openclFree(d_b);
   openclFree(d_c);
   for(i=0; i<iNumElements; ++i){
      if(h_c[i] != h_a[i]+h_b[i]){
        ++ierr;
        printf("h_c[%d]=%f ,should be %f (%f)+(%f)\n",i,h_c[i],h_a[i]+h_b[i],h_a[i],h_b[i]);
      }
   }
   if(ierr == 0){
      printf("pass\n");
   }
   free(h_a);
   free(h_b);
   free(h_c);
   return 0;
}
