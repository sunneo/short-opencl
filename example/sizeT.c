#include "opencl_runtime.h"
#include <stdlib.h>
#include <stdio.h>
int main(){
   size_t* sizeptr;
   size_t* sizeptr_h;
   int size = 1024;
   size_t dimglobal[3] = {8192,1,1};
   size_t dimlocal[3] = {256,1,1};
   int i;
   openclInitFromFile("sizeT.cl");
   openclMalloc((void**)&sizeptr,sizeof(size_t)*size);
/*   
   openclSetArgument(&sizeptr,sizeof(size_t*),0);
   openclSetArgument(&size,sizeof(int),1);
   openclConfigureCall(dimlocal,dimglobal);
   openclLaunch("kernelfnc");
*/
   openclLaunchKernel("kernelfnc",dimlocal,dimglobal,sizeptr,size);
   openclThreadSynchronize();
   sizeptr_h = (size_t*)malloc(sizeof(size_t)*size);
   openclMemcpy(sizeptr_h,sizeptr,sizeof(size_t)*size, openclMemcpyDeviceToHost);
   openclFree(sizeptr);
   for(i=0; i<size; ++i){
      printf("%-4zd ",sizeptr_h[i]);
   }
   printf("\n");
   free(sizeptr_h);
   return 0;
}
