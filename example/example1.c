#include <stdio.h>
#include <stdlib.h>
#include "opencl_runtime.h"

int main(){
   int size = 1024;
   int i;
   int dimLocal[3]={256,1,1};
   int dimGlobal[3]={1024,1,1};
   const char* src = "\n\
__kernel void vecAdd(int __global* a, int __global* b, int __global* c,unsigned size){\n\
   int id = get_global_id(0);  \n\
   if(id < size){\n\
      c[id]=a[id]+b[id];\n\
   }\n\
}";
   int *dA,*dB,*dC;
   int *hA,*hB,*hC;
   hA =  (int*)malloc(sizeof(int)*size);
   hB =  (int*)malloc(sizeof(int)*size);
   hC =  (int*)malloc(sizeof(int)*size);
   for(i=0; i<size; ++i){
      hA[i] = 1;
      hB[i] = 2;
   }
   openclInitFromSource(src);
   openclMalloc((void**)&dA,sizeof(int)*size);
   openclMalloc((void**)&dB,sizeof(int)*size);
   openclMalloc((void**)&dC,sizeof(int)*size);
//   openclMemset(dA,0,sizeof(int)*size);
//   openclMemset(dB,0,sizeof(int)*size);
   openclMemset(dC,0,sizeof(int)*size);



   openclMemcpy(dA,hA,sizeof(int)*size,openclMemcpyHostToDevice);
   openclMemcpy(dB,hB,sizeof(int)*size,openclMemcpyHostToDevice);
   openclLaunchGrid("vecAdd",dimLocal,dimGlobal,dA,dB,dC,size);
   openclThreadSynchronize();
   
   openclMemcpy(hC,dC,sizeof(int)*size,openclMemcpyDeviceToHost);
   for(i=0; i<size; ++i){
      printf("%d",hC[i]);
   }
   printf("\n");

   openclMemset(dC,0,sizeof(int)*size);
   openclMemcpy(hC,dC,sizeof(int)*size,openclMemcpyDeviceToHost);

   for(i=0; i<size; ++i){
      printf("%d",hC[i]);
   }
   printf("\n");

   openclFree(dA);
   openclFree(dB);
   openclFree(dC);
   free(hA);
   free(hB);
   free(hC);
   return 0;
}
