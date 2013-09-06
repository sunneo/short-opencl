#include <stdlib.h>
#include <stdio.h>
#include "opencl_runtime.h"



static size_t getKernelPtrAddr(void* clmemaddr){
   static openclCtx ctx;
   const static char* __kernel_ptr_getter="\n\
__kernel void getKernelPtrAddr(unsigned __global* output,unsigned __global* input){ \n\
   *output = (unsigned)input;\n\
}\n\
";
   int localdim[3]={1,1,1};
   int globaldim[3]={1,1,1};
   size_t ret = 0;
   size_t *paddr;
   if(!ctx){
      openclCtx current;
      openclCtxPopCurrent(&current);
      ctx = openclCtxCreateFrom(current);
      openclInitFromSource2(ctx,__kernel_ptr_getter);
   }
   openclMalloc2(ctx,(void**)&paddr,sizeof(size_t));
   openclLaunchGrid2(ctx,"getKernelPtrAddr",localdim,globaldim,paddr,clmemaddr);
   openclMemcpy2(ctx,&ret,paddr,sizeof(size_t),openclMemcpyDeviceToHost);
   openclFree2(ctx,paddr);
   return ret;
}
typedef struct StructWithoutPointerButAddress{
   unsigned addr0;
   unsigned  addr1;
}StructWithoutPointerButAddress;


int main(){
   unsigned* memValue;
   unsigned* offsetMemValue;
   StructWithoutPointerButAddress* strPtr;
   int offsetValue = 1;
   int hostValue[3];
   int localdim[3]={256,1,1};
   int globaldim[3]={1024,1,1};
   int i;
   size_t kerneladdr_memvalue;
   int devCnt,platCnt;
   int currentDev,currentPlatform;
   openclGetDeviceCount(&devCnt);
   openclGetPlatformCount(&platCnt);

   printf("there are %d devices, %d platforms\n",devCnt,platCnt);
   openclGetDevice(&currentDev);
   openclGetPlatform(&currentPlatform);
   printf("current platform %d, current dev %d\n",currentPlatform, currentDev);
   openclSetDevice(0,0);
   
   openclInitFromFile("kernel.cl"); 
   /// allocate a bundle of buffer.
   openclMalloc((void**)&memValue,sizeof(int)*4);
   openclMalloc((void**)&strPtr,sizeof(StructWithoutPointerButAddress));
   kerneladdr_memvalue = getKernelPtrAddr(memValue);
   openclMemcpy((void*)&strPtr->addr0,&kerneladdr_memvalue,sizeof(size_t),openclMemcpyHostToDevice);
   /// alternatively, you may try to invoke a kernel in line like cuda-runtime style
   openclLaunchGrid("kernelFnc",localdim,globaldim,memValue,memValue,memValue,offsetValue,strPtr);

   /// wait for kernel finished.
   openclThreadSynchronize();
   /// copy data from allocated one to the host one.
   openclMemcpy(hostValue,memValue,sizeof(int)*4,openclMemcpyDeviceToHost);
   /// print it out.
   for(i=0; i<4; ++i){
      printf("mem[%d]=%x\n",i,hostValue[i]);
   }
   /// release allocated memory object.
   openclFree(memValue);
   return 0;
}
