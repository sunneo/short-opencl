#include <stdlib.h>
#include <stdio.h>
#include "opencl_runtime.h"
int main(){
   unsigned* memValue;
   unsigned* offsetMemValue;
   int offsetValue = 1;
   int hostValue[3];
   int localdim[3]={256,1,1};
   int globaldim[3]={1024,1,1};
   int i;
   /// initialize program with a source code.
   /// user can use openclInitFromSource(SourceString) as an alternation.
   openclInitFromFile("kernel.cl"); 
   /// allocate a bundle of buffer.
   openclMalloc((void**)&memValue,sizeof(int)*4);
   /// get a shifted pointer (subbuffer) from an allocated buffer.
   openclShiftPointer((void**)&offsetMemValue,memValue,4);
   /// user can invoke a kernel by configure call, setargument, and then launch it.
/*   
   openclConfigureCall( localdim, globaldim);
   openclSetArgument(&memValue,sizeof(unsigned*),0);
   openclSetArgument(&memValue,sizeof(unsigned*),1);
   openclSetArgument(&offsetMemValue,sizeof(unsigned*),2);
   openclSetArgument(&offsetValue,sizeof(int),3);
   openclLaunch("kernelFnc");
*/
   /// alternatively, you may try to invoke a kernel in line like cuda-runtime style
   openclLaunchGrid("kernelFnc",localdim,globaldim,&memValue,&memValue,&offsetMemValue,&offsetValue);
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
