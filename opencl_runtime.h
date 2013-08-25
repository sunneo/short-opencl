#ifndef OPENCL_RUNTIME_H_
#  define OPENCL_RUNTIME_H_ 1
#include <CL/opencl.h>

struct openclPlatformInfo;
typedef struct openclDeviceInfo{
   cl_device_id deviceid;
   cl_uint addressBits;
   cl_bool isDeviceAvailable;
   cl_bool isDeviceCompilerAvailable;
   cl_device_fp_config deviceDoubleFPConfig;
   cl_bool isLittleEndian;
   cl_bool isErrorCorrectionSupport;
   cl_device_exec_capabilities executionCapabilities;
   char extensions[1024];
   cl_ulong global_mem_cache_size;
   cl_uint      global_mem_cache_type;
   cl_uint      global_mem_cacheline_size;
   cl_ulong global_mem_size;
   cl_ulong local_mem_size;
   cl_device_local_mem_type  local_mem_type;
   cl_uint maxClockFreq;
   cl_uint maxComputeUnits;
   cl_uint maxConstantArgs;
   cl_ulong maxConstantBufferSize;
   cl_ulong maxMemAllocSize;
   size_t maxParamSize;
   size_t maxWorkGroupSize;
   size_t maxWorkItemDims;
   size_t maxWorkItemSizes[3];
   cl_uint memBaseAddrAlign;
   cl_uint minDataTypeAlignSize;
   char deviceName[256];
   struct openclPlatformInfo* platform;
   cl_device_type deviceType;
   char vendor[256];
   cl_uint vendorID;
   cl_uint version;
   char deviceVersion[32];
   char driverVersion[32];
}openclDeviceInfo;

typedef struct openclPlatformInfo{
   cl_platform_id platformid;
   int deviceCount;
   openclDeviceInfo* deviceInfos;
   char platformName[256];
   char vendorName[256];
   char profile[256];
   char version[32];
   char extensions[1024];
}openclPlatformInfo;

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
 * set a block of devicePointer to a value, just like memset in string.h
 * @param dstPtr pointer to block of memory object
 * @param bytevalue 
 * @param size size of block
 */
int  openclMemset(void* dstPtr,int bytevalue, size_t size);
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

/**
 * get device count of current context
 * default is in GPU type
 * it would return the device count in the first platform default.
 * @param count device count to return.
 */
void openclGetDeviceCount(int* count);
/**
 * get total count of platforms.
 * @param count platform count to return.
 */
void openclGetPlatformCount(int* count);
/**
 * get the platform which is being using.
 * @param platform platform index to get. 
 */
void openclGetPlatform(int* platform);
/**
 * get the device being using.
 * @param device pointer to a return value. 
 */
void openclGetDevice(int* device);
/**
 * get properties of the current platform
 * @param info pointer to platform info structure.
 */
void openclGetPlatformProperties(openclPlatformInfo* info);
/** 
 * get properties of current device.
 * @param info pointer to device info structure.
 */ 
void openclGetDeviceProperties(openclDeviceInfo* info);


typedef void *openclCtx;
/**
 * create an empty context
 * user can use it to create a new context for another platform and device.
 * 
 */
openclCtx openclCtxCreate();
/**
 * create a context which inherits another context.
 * user can simplely use it to attach to a existent context, and
 * initialize it with new source.
 * @param c another context.
 */
openclCtx openclCtxCreateFrom(openclCtx c);
/**
 * destroy a context
 * @param c context to destroy.
 */
void openclCtxDestroy(openclCtx c);

/**
 * push a context into current work thread
 * set the context as current context.
 * @param c context 
 */
void openclCtxPushCurrent(openclCtx c);
/**
 * peek the current working context
 * @param c pointer to a returned value.
 */
void openclCtxPeekCurrent(openclCtx* c);
/**
 * pop current context to the returned value and 
 * set the current context to the default one.
 * @param c pointer to a returned value.
 */
void openclCtxPopCurrent(openclCtx* c);

void openclSetDevice(int platform,int device);

void openclSetDevice2(openclCtx c,int platform,int device);
void openclInitFromSource2(openclCtx c,const char* src);
void openclInitFromFile2(openclCtx c,const char* path);
int  openclMalloc2(openclCtx c,void** ptr,size_t size);
int  openclMemset2(openclCtx c,void* dst,int bytevalue, size_t size);
int  openclMemcpy2(openclCtx c,void* dst, const void* src, size_t size, openclMemcpyKind kind);
int  openclFree2(openclCtx c,void* ptr);
int  openclThreadSynchronize2(openclCtx c);
void openclSetArgument2(openclCtx c,void* arg, size_t size, size_t idx);
int  openclConfigureCall2(openclCtx c,size_t localdim[3],size_t globaldim[3]);
void openclLaunch2(openclCtx c,const char* kernel);
void openclLaunchGrid2(openclCtx c,const char* kernel,size_t localdim[3],size_t globaldim[3],...);

void openclGetDeviceCount2(openclCtx openclctx,int* count);
void openclGetPlatformCount2(openclCtx openclctx,int* count);
void openclGetPlatform2(openclCtx openclctx,int* platform);
void openclGetDevice2(openclCtx openclctx,int* device);
void openclGetPlatformProperties2(openclCtx openclctx,openclPlatformInfo* info);
void openclGetDeviceProperties2(openclCtx openclctx,openclDeviceInfo* info);

#endif
