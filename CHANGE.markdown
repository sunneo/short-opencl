* 2013-08-16 support offset to pointer(memory object) by runtime.
* 2013-08-16 change behaviors of openclLaunchGrid, parameters are passed by values.
    int* memValue;
    int hostValue[4];
    int i;
    openclMalloc((void**)&memValue,sizeof(int)*4);
    openclLaunchGrid("kernelFnc",localdim,globaldim,memValue,memValue,memValue+1,offsetValue);
    //when we pass an shifted pointer, it would be handled in the runtime.
    openclMemcpy(hostValue,memValue,sizeof(int)*4,openclMemcpyDeviceToHost);
    for(i=0; i<4; ++i){
        printf("mem[%d]=%x\n",i,hostValue[i]);
    }
