#ifndef OPENCL_RUNTIME_H_
#  define OPENCL_RUNTIME_H_ 1


enum openclMemcpyKind{
   openclMemcpyHostToDevice,
   openclMemcpyDeviceToHost,
   openclMemcpyHostToHost,
   openclMemcpyDeviceToDevice
};
typedef enum openclMemcpyKind openclMemcpyKind;

/**
 * initialize opencl runtime from a character array with source program.
 * @param src a pointer points to source.
 */
void openclInitFromSource(const char* src);

/**
 * initialize opencl runtime from a path to kernel program.
 * @param src path to kernel source.
 */
void openclInitFromFile(const char* path);
/**
 * allocate read-write memory object in size.
 * the allocated memory object would be return to the ptr parameter.
 * @param ptr  pointer to get returned object.
 * @param size size of memory object.
 */
int  openclMalloc(void** ptr,size_t size);
/**
 * get an shifted pointer from an allocated pointer.
 * this is implemented with an OpenCL API clCreateSubBuffer
 * the new one would inherts flags from the source memory object.
 * @param ptr pointer to the return object.
 * @param srcPtr the original object.
 * @param offset shift amount from the srcPtr.
 */
int  openclShiftPointer(void** ptr,const void* srcPtr, size_t offset);
/**
 * copy memory between pointer and memory object.
 * kind: openclMemcpyDeviceToHost 
 * kind: openclMemcpyDeviceToDevice copy from one memory to another memory object.
 * kind: openclMemcpyHostToDevice copy from host to device.
 * kind: openclMemcpyHostToHost copy from source to destination. (adopts built-in memcpy)
 * @param dst destinatation memory object/address
 * @param src source memory object/address
 * @param size count of block to copy
 * @param kind direction of data-transferring.
 */
int  openclMemcpy(void* dst, const void* src, size_t size, openclMemcpyKind kind);
/**
 * release allocated memory object.
 * @param ptr object to release.
 */
int  openclFree(void* ptr);
/** 
 * synchronize to launched kernel.
 */
int  openclThreadSynchronize();
/**
 * set argument to the kernel.
 * @param arg pointer to argument.
 * @param size size of the argument
 * @param idx index of the argument.
 */
void openclSetArgument(void* arg, size_t size, size_t idx);
/**
 * set up configuration of kernel. 
 * @param localdim  dimension of a local threadgroup(working item group)
 * @param globaldim dimension of global threadgroup
 */
int  openclConfigureCall(size_t localdim[3],size_t globaldim[3]);
/**
 * launch a kernel.
 * @param kernel name of the kernel function.
 */
void openclLaunch(const char* kernel);
/**
 * launch a kernel 
 * user could use this function to straight-forward invoke a kernel in a line.
 * arguments are passed as pointer in va_arg.
 * @param kernel name of the kernel function.
 * @param localdim dimension of a local threadgroup.
 * @param globaldim dimension of global threadgroup.
 * 
 */
void openclLaunchGrid(const char* kernel,size_t localdim[3],size_t globaldim[3],...);

typedef void *openclCtx;
openclCtx openclCtxCreate();
openclCtx openclCtxCreateFrom(openclCtx c);
void openclCtxDestroy(openclCtx c);
void openclCtxPushCurrent(openclCtx c);
void openclCtxPeekCurrent(openclCtx* c);
void openclCtxPopCurrent(openclCtx* c);
void openclInitFromSource2(openclCtx c,const char* src);
void openclInitFromFile2(openclCtx c,const char* path);
int  openclMalloc2(openclCtx c,void** ptr,size_t size);
int  openclShiftPointer2(openclCtx c,void** ptr,const void* srcPtr, size_t offset);
int  openclMemcpy2(openclCtx c,void* dst, const void* src, size_t size, openclMemcpyKind kind);
int  openclFree2(openclCtx c,void* ptr);
int  openclThreadSynchronize2(openclCtx c);
void openclSetArgument2(openclCtx c,void* arg, size_t size, size_t idx);
int  openclConfigureCall2(openclCtx c,size_t localdim[3],size_t globaldim[3]);
void openclLaunch2(openclCtx c,const char* kernel);
void openclLaunchGrid2(openclCtx c,const char* kernel,size_t localdim[3],size_t globaldim[3],...);
#endif
