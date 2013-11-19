
* 2013-11-20 add macros to support varidiac argument
i.e in example/VectorAdd.c

kernel function can be invoked by openclLaunchKernel simply.
    openclLaunchKernel("VectorAdd",dimglobal,dimlocal,d_a,d_b,d_c,iNumElements);

which would be expanded into:
<pre>
    {                                                               
        int __openclkernelLaunchNarg = 0;                           
        openclLaunchGridConfigureCall(dimglobal,dimlocal);      
        openclLaunchGridSetArg(&da,sizeof(da),__openclkernelLaunchNarg++);               
        openclLaunchGridSetArg(&db,sizeof(db),__openclkernelLaunchNarg++);               
        openclLaunchGridSetArg(&dc,sizeof(dc),__openclkernelLaunchNarg++);               
        openclLaunchGridSetArg(&iNumElements,sizeof(iNumElements),__openclkernelLaunchNarg++);               
        openclLaunch("VectorAdd");
    }
</pre>




* 2013-08-16 support offset to pointer(memory object) by runtime.
* 2013-08-16 change behaviors of openclLaunchGrid, parameters are passed by values.

    
i.e. in the example/launcher.c :

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


the result would show out four line of the same value.

    mem[0]=21004
    mem[1]=21004
    mem[2]=21004
    mem[3]=21004
