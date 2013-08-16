#include <CL/opencl.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/stat.h>
#include <string.h>
#include "utils/vector.h"
#include "opencl_runtime.h"
typedef struct MemObjRecord{
   void* start;
   size_t len;
}MemObjRecord;

static Vector* memObjList;

static Vector* mem_obj_list_get();
static void mem_obj_list_clear();
static void mem_obj_list_remove(void* key);
static MemObjRecord* mem_obj_list_get_hit(void* key);

static MemObjRecord* mem_obj_new(void* s,size_t l);

static Vector* mem_obj_list_get(){
   if(memObjList == NULL){
      memObjList = vector_create(0);
   }
   return memObjList;
}

static void mem_obj_delete(MemObjRecord* o){
   if(o != NULL){
      clReleaseMemObject((cl_mem)o->start);
      free(o);
   }
}

static void mem_obj_list_clear(){
    Vector* vec;
    vec = mem_obj_list_get();
    vector_foreach(vec,(void(*)(void*))mem_obj_delete);
    vector_clear(vec);
}

static void mem_obj_list_remove(void* key){
    Vector* vec;
    int i,size;
    vec = mem_obj_list_get();
    size = vector_size(vec);
    for(i=0; i<size; ++i){
       MemObjRecord* o = (MemObjRecord*)vector_at(vec,i);
       if(o->start == key){
          break;
       }
    }
    if(i < size){
       vector_erase(vec,i);
    }
}

static MemObjRecord* mem_obj_list_get_hit(void* key){
    Vector* vec;
    int i,size;
    vec = mem_obj_list_get();
    size = vector_size(vec);
    for(i=0; i<size; ++i){
       MemObjRecord* o = (MemObjRecord*)vector_at(vec,i);
       if((size_t)key >=((size_t)o->start) && 
            (size_t)key <((size_t)o->start)+o->len){
          return o;
       }
    }
    return NULL;
}

static MemObjRecord* mem_obj_new(void* s,size_t l){
   MemObjRecord* ret;
   ret = (MemObjRecord*)malloc(sizeof(MemObjRecord));
   ret->start = s;
   ret->len = l;
   return ret;
}

static int mem_obj_CL_INVALID_MEM_OBJECT_handler(int _err,const void* dst,void** ptr){
   if(_err == CL_INVALID_MEM_OBJECT){
      MemObjRecord* o = mem_obj_list_get_hit((void*)dst);
      size_t offset = ((size_t)dst) - (size_t)o->start;
      void* shiftPtr;
      openclShiftPointer(&shiftPtr,o->start,offset);
      *ptr = shiftPtr;
      return 1; //handled
   }    
   return 0;
}

static int mem_obj_CL_INVALID_MEM_OBJECT_handler_with_orig_and_offset(
   int _err,const void* dst,void** ptr,size_t* r_offset
){
   if(_err == CL_INVALID_MEM_OBJECT){
      MemObjRecord* o = mem_obj_list_get_hit((void*)dst);
      size_t offset = ((size_t)dst) - (size_t)o->start;
      *ptr = o->start;
      *r_offset = offset;
      return 1; //handled
   }    
   return 0;
}

static void printLastError(int _err)
{
   switch(_err)
   {
         default:
                fprintf(stderr,"Error Code=%d\n",_err);
                break;
         case CL_INVALID_MEM_OBJECT:
				fprintf(stderr,"Invalie Memory Object\n");
                break;
         case CL_INVALID_KERNEL_ARGS:
                fprintf(stderr,"Invalid Kernel Arguments\n");
                break;
         case CL_SUCCESS:
                fprintf(stderr,"Success\n");
                break;
         case CL_INVALID_DEVICE:
                fprintf(stderr,"Invalid Device \n");
                break;
         case CL_INVALID_PLATFORM:
                fprintf(stderr,"Invalid Platform\n");
                break;
         case CL_INVALID_DEVICE_TYPE:
                fprintf(stderr,"Invalid Device Type\n");
                break;
         case CL_DEVICE_NOT_FOUND:
                fprintf(stderr,"Device Not Found\n");
                break;
         case CL_INVALID_CONTEXT:
                fprintf(stderr,"Invalid Context \n");
                break;
         case CL_INVALID_VALUE:
                fprintf(stderr,"Invalid Value\n");
                break;
         case CL_INVALID_BUFFER_SIZE:
                fprintf(stderr,"Invalid Buffer Size \n");
                break;
         case CL_INVALID_HOST_PTR:
                fprintf(stderr,"Invalid Host Pointer \n");
                break;
         case CL_MEM_OBJECT_ALLOCATION_FAILURE:
                fprintf(stderr,"Memory Object Allcation Failure\n");
                break;
         case CL_INVALID_QUEUE_PROPERTIES:
                fprintf(stderr,"Invalid Queue Properties\n");
         case CL_OUT_OF_RESOURCES:
                fprintf(stderr,"Out of Resource\n");
                break;
         case CL_OUT_OF_HOST_MEMORY:
                fprintf(stderr,"Out of Host Memory\n");
                break;
         case CL_INVALID_COMMAND_QUEUE:
                fprintf(stderr,"Invalid Command Queue\n");
                break;
         case CL_INVALID_EVENT_WAIT_LIST:
                fprintf(stderr,"Invalid Event Wait List\n");
                break;
         case CL_MISALIGNED_SUB_BUFFER_OFFSET:
                fprintf(stderr,"MisAligned Sub Buffer Offset\n");
                break;
         case CL_MEM_COPY_OVERLAP:
                fprintf(stderr,"Memory Copy Overlapped\n");
                break;
   } 
}

static void OMPCLCreateCommandQueue(cl_command_queue* cmdqueue,
                cl_context ctx,cl_device_id devid){
   int err;
   *cmdqueue = clCreateCommandQueue(ctx, devid, 0, &err);
#if __ompclDebug != 0
   {
         if(err != CL_SUCCESS){
                fprintf(stderr, "Error in clCreateCommandQueue(OMPCLCreateCommandQueue)\n");                                                                                              
                printLastError(err);
                exit(0);
         }
   }
#endif
}

static void OMPCLGetPlatformID(cl_platform_id* id,cl_device_type type)
{
   int err;
   char cBuffer[1024];
   int ciDeviceCount;
   int clPlatformCnt;
   cl_platform_id* ids;
   int i;
   clGetPlatformIDs(0,NULL,&clPlatformCnt);
   ids = (cl_platform_id*)malloc(sizeof(cl_platform_id)*clPlatformCnt);
   err = clGetPlatformIDs(clPlatformCnt,ids,&clPlatformCnt);
   for(i=0; i<clPlatformCnt; ++i)
   {
         clGetDeviceIDs (ids[i], type, 0, NULL, &ciDeviceCount);
         clGetPlatformInfo(ids[i], CL_PLATFORM_NAME, sizeof(cBuffer), cBuffer, NULL);
         if(ciDeviceCount > 0)
         {
                *id = ids[i];
                break;
         }
   }

#if __ompclDebug != 0
   {
         if(err != CL_SUCCESS){
                fprintf(stderr, "Error in clGetPlatformIDs(OMPCLGetPlatformID)\n");
                printLastError(err);
                exit(0);
         }
   }
#endif
}
       
static void OMPCLGetDeviceID(cl_device_id* devid,cl_platform_id platformid,cl_device_type type)
{
   int err;
   err = clGetDeviceIDs(platformid,type,1,devid,NULL);
#if __ompclDebug != 0
   {
         if(err != CL_SUCCESS){
                fprintf(stderr, "Error in clGetDeviceIDs(OMPCLGetDeviceID)\n");
                printLastError(err);
                exit(0);
         }
   }
#endif
}

static void OMPCLCreateContext(cl_context* ctx,cl_device_id devid)
{
   int _err;
   *ctx =  clCreateContext(NULL, 1, &devid, NULL, NULL,&_err);
#if __ompclDebug != 0
   {
         if(_err != CL_SUCCESS){
                fprintf(stderr, "Error in clCreateContext(OMPCLCreateContext)\n");
                printLastError(_err);
                exit(0);
         }
   }
#endif
}

static void OMPCLInit(
                cl_device_type type,
                cl_platform_id* platformid,
                cl_device_id* devid,
                cl_context* ctx,
                cl_command_queue* cmdqueue,
                const char* srcCode,
                cl_program* program,
                int* compiled)
{
/*   printf(
         "OMPCLInit(%d,&platformid=%x,&devid=%x,&ctx=%x,&cmdqueue=%x,src=%x,&program=%x,compiled=%d)\n",
         type,platformid,devid,ctx,cmdqueue,srcCode,program,*compiled);*/
   //printf("before platform id OMPCLInit::output(%d,platformid=%x,devid=%x,ctx=%x,cmdqueue=%x,program=%p)\n", type,*platformid,*devid,*ctx,*cmdqueue,*program);

   if(*compiled != 1){
      *compiled = 1;
      OMPCLGetPlatformID(platformid,type);
      //printf("after get platform id OMPCLInit::output(%d,platformid=%x,devid=%x,ctx=%x,cmdqueue=%x,program=%x)\n", type,*platformid,*devid,*ctx,*cmdqueue,*program);

         OMPCLGetDeviceID(devid,*platformid,type);
      //printf("after get dev id OMPCLInit::output(%d,platformid=%x,devid=%x,ctx=%x,cmdqueue=%x,program=%x)\n", type,*platformid,*devid,*ctx,*cmdqueue,*program);
         OMPCLCreateContext(ctx,*devid);
      //printf("after get ctx OMPCLInit::output(%d,platformid=%x,devid=%x,ctx=%x,cmdqueue=%x,program=%x)\n", type,*platformid,*devid,*ctx,*cmdqueue,*program);
         {
             char* prog = (char*)srcCode;
             size_t len = (size_t)strlen(prog);
             const char buildOptions[] = "-w";
             int errret = 0;
             *program = clCreateProgramWithSource(*ctx,1,(const char**)&prog,&len,&errret);
             if(errret != CL_SUCCESS){
                fprintf(stderr, "Error in OMPCLInit(clCreateProgramWithSource)\n");
                printLastError(errret);
                exit(0);
             }
             errret = clBuildProgram(*program,1,devid,buildOptions,NULL,NULL);
             if (errret != CL_SUCCESS) {
                int status;
                char* programLog;
                size_t logSize=4096;
                cl_device_id device = devid[0];
                // check build error and build status first
                clGetProgramBuildInfo(*program, device, CL_PROGRAM_BUILD_STATUS,sizeof(cl_build_status), &status, NULL);
                clGetProgramBuildInfo(*program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
                programLog = (char*) calloc (logSize+1, sizeof(char));
                clGetProgramBuildInfo(*program, device, CL_PROGRAM_BUILD_LOG, logSize+1, programLog, NULL);
                printf("Build failed; error=%d, status=%d, programLog:\n\n%s", errret, status, programLog);
                free(programLog);
                exit(0);
             }
         }
      //printf("after get program OMPCLInit::output(%d,platformid=%x,devid=%x,ctx=%x,cmdqueue=%x,program=%x)\n", type,*platformid,*devid,*ctx,*cmdqueue,*program);
      OMPCLCreateCommandQueue(cmdqueue,*ctx,*devid);
      //printf("after get cmdqueue OMPCLInit::output(%d,platformid=%x,devid=%x,ctx=%x,cmdqueue=%x,program=%x)\n", type,*platformid,*devid,*ctx,*cmdqueue,*program);


   }
   //printf("after all OMPCLInit::output(%d,platformid=%x,devid=%x,ctx=%x,cmdqueue=%x,program=%x)\n", type,*platformid,*devid,*ctx,*cmdqueue,*program);

}
static void OMPCLCreateBuffer(cl_mem* mem,cl_context ctx,size_t size){
   //printf("OMPCLCreateBuffer(%p,%p,%u)\n",mem,ctx,size);
   int err;
   *mem = clCreateBuffer(ctx, CL_MEM_READ_WRITE, size, NULL,&err);
#if __ompclDebug != 0
   {
         if(err != CL_SUCCESS){
                fprintf(stderr, "Error in clCreateBuffer(OMPCLCreateBuffer)\n");                                                                                                          
                printLastError(err);
                exit(0);
         }
   }
#endif
}


typedef struct OpenCLRuntimeAPI{
  char* launchFnc;
  size_t workdim;
  size_t localdim[3];
  size_t globaldim[3];
  struct {
     void* argList[ 256 ];
     size_t argSize[ 256 ];
     size_t argIdxes[ 256 ];
     size_t argIdx;
  } setupArg;
  cl_platform_id ompclPlatformID;
  cl_device_id ompclDeviceID;
  cl_context ompclContext;
  cl_command_queue ompclCommandQueue;
  cl_program ompclProgram;
  int ompclCompiled; 
  int inited;
}OpenCLRuntimeAPI;

static OpenCLRuntimeAPI openclRuntime;

static void openclRuntimeReleaseOnExit(){
   clReleaseCommandQueue(openclRuntime.ompclCommandQueue);
   clReleaseContext(openclRuntime.ompclContext);
   clReleaseProgram(openclRuntime.ompclProgram);
   clUnloadCompiler();
}


void openclInitFromSource(const char* src){
   OMPCLInit(CL_DEVICE_TYPE_GPU,
      &openclRuntime.ompclPlatformID,&openclRuntime.ompclDeviceID,&openclRuntime.ompclContext, 
      &openclRuntime.ompclCommandQueue,src, &openclRuntime.ompclProgram, &openclRuntime.ompclCompiled );
   atexit(openclRuntimeReleaseOnExit);
}

void openclInitFromFile(const char* file){
   struct stat s;
   cl_program ret = 0;
   if(openclRuntime.inited){
      return;
   }
   if(stat(file,&s) != 0){
      perror(file);
      exit(0);
   }
   {
      char* srcCode = (char*)malloc((size_t)s.st_size);
      {
         FILE* fp = fopen(file,"r");
         if(!fp)
         {
            perror(file);
            exit(0);
         }
         fread(srcCode,1,(size_t)s.st_size,fp);
         fclose(fp);
         if(!openclRuntime.inited)
         {
            openclRuntime.inited=1;
            openclInitFromSource(srcCode);
         }
      }
   }
}

static int openclCheckInited(){
   if(!openclRuntime.inited){
      fprintf(stderr,"OpenCL not inited\n");
      return 0;
   }
   return 1;
}

int openclMalloc(void** ptr,size_t size){
    cl_mem ret;
    int err;
    if(!openclCheckInited()) return -1;
    ret = clCreateBuffer(openclRuntime.ompclContext, CL_MEM_READ_WRITE, size, NULL,&err);
    vector_push_back(mem_obj_list_get(),mem_obj_new((void*)ret,size));
    *ptr = (void*)ret;
    return err;
}

int openclMemcpy(void* dst, const void* src, size_t size, openclMemcpyKind kind){
   if(!openclCheckInited()) return -1;
   switch(kind){
      default:
      {
         fprintf(stderr,"Unknown openclMemcpyKind:%d\n",kind);
         return -1;
      }
      case openclMemcpyHostToDevice:
      {
          cl_int err;
          void* dstPtr;
          size_t offset;
          err = clEnqueueWriteBuffer(openclRuntime.ompclCommandQueue,(cl_mem)dst,CL_TRUE,0,size,src,0,0,0);
          if(mem_obj_CL_INVALID_MEM_OBJECT_handler_with_orig_and_offset(err,dst,&dstPtr,&offset)){
             err = clEnqueueWriteBuffer(openclRuntime.ompclCommandQueue,(cl_mem)dstPtr,CL_TRUE,offset,size,src,0,0,0);
             clReleaseMemObject((cl_mem)dstPtr);
          }
          return err;
      }
      case openclMemcpyDeviceToHost:
      {
          cl_int err;
          void* srcPtr;
          size_t offset;
          err = clEnqueueReadBuffer(openclRuntime.ompclCommandQueue,(cl_mem)src,CL_TRUE,0,size,dst,0,0,0);
          if(mem_obj_CL_INVALID_MEM_OBJECT_handler_with_orig_and_offset(err,src,&srcPtr,&offset)){
             err = clEnqueueReadBuffer(openclRuntime.ompclCommandQueue,(cl_mem)srcPtr,CL_TRUE,offset,size,dst,0,0,0);
          }
          return err;
      }
      case openclMemcpyHostToHost:
      {
         memcpy(dst,src,size);
         return 0;
      }
      case openclMemcpyDeviceToDevice:
      {
          cl_int err;
          void* dstPtr;
          void* srcPtr; 
          size_t dstOffset;
          size_t srcOffset;
          err = clEnqueueCopyBuffer(openclRuntime.ompclCommandQueue,(cl_mem)src,(cl_mem)dst,0,0,size,0,0,0);
          if(err == CL_INVALID_MEM_OBJECT){
             mem_obj_CL_INVALID_MEM_OBJECT_handler_with_orig_and_offset(err,src,&srcPtr,&srcOffset);
             mem_obj_CL_INVALID_MEM_OBJECT_handler_with_orig_and_offset(err,dst,&dstPtr,&dstOffset);
             err = clEnqueueCopyBuffer(openclRuntime.ompclCommandQueue,(cl_mem)srcPtr,(cl_mem)dstPtr,srcOffset,dstOffset,size,0,0,0);
          }
          return err;
      }
   }

}
int openclFree(void* ptr){
   if(!openclCheckInited()) return -1;
   return clReleaseMemObject((cl_mem)ptr);
}
int openclThreadSynchronize(){
   if(!openclCheckInited()) return -1;
   return clFinish(openclRuntime.ompclCommandQueue);
}

void openclSetArgument(void* arg, size_t size, size_t index){
   if(!openclCheckInited()) return ;
   if(openclRuntime.setupArg.argIdx >= 256){
      fprintf(stderr,"Count of Argument exceeds 256\n");
      exit(0);
   }
   openclRuntime.setupArg.argList[openclRuntime.setupArg.argIdx]=(void*)malloc(size);
   memcpy(openclRuntime.setupArg.argList[openclRuntime.setupArg.argIdx],arg,size);
   openclRuntime.setupArg.argSize[openclRuntime.setupArg.argIdx]=size;
   openclRuntime.setupArg.argIdxes[openclRuntime.setupArg.argIdx]=index;
   ++openclRuntime.setupArg.argIdx;
}

int openclConfigureCall(size_t localdim[3],size_t globaldim[3]){
   size_t workdim = 3;
   int idx;
   if(!openclCheckInited()) return -1; 
   for(idx = 2; idx >0; --idx){
     if(localdim[idx] <= 1 || globaldim[idx]  <= 1)
       --workdim;
   }
   for(idx=0; idx<3; ++idx){
      openclRuntime.localdim[idx] = localdim[idx];
      openclRuntime.globaldim[idx] = globaldim[idx];
   }
   openclRuntime.workdim = workdim;
   if(workdim == 0){
      return -1;
   }
   return 0;
}

static void vec_release_list_releaseMemObj(void* o){
   clReleaseMemObject((cl_mem)o);
}

static void openclLaunchKernelObject(cl_kernel kernel,const char* kernelName){
   int argCfg;
   int err;
   Vector* vecReleaseList = vector_create(0);
   for(argCfg=0; argCfg<openclRuntime.setupArg.argIdx; ++argCfg){
      err = clSetKernelArg(
           kernel,
           openclRuntime.setupArg.argIdxes[argCfg],
           openclRuntime.setupArg.argSize[argCfg],
           openclRuntime.setupArg.argList[argCfg]
      );
      /*printf("setKernelArg (%x,idx=%d,size=%d,value=%x)\n",kernel,
          openclRuntime.setupArg.argIdxes[argCfg],
          openclRuntime.setupArg.argSize[argCfg],
          *(void**)openclRuntime.setupArg.argList[argCfg]
      );*/
      if(err != 0){
         if(err == CL_INVALID_MEM_OBJECT){
             void* ptr;
             
             mem_obj_CL_INVALID_MEM_OBJECT_handler(
                 err,
                 *((void**)openclRuntime.setupArg.argList[argCfg]),
                 &ptr
             );
             vector_push_back(vecReleaseList,ptr);
             err = clSetKernelArg(
               kernel,
               openclRuntime.setupArg.argIdxes[argCfg],
               openclRuntime.setupArg.argSize[argCfg],
               &ptr
             );
         }
         if(err != 0){
            fprintf(stderr,"uncaught exception: Error while setup kernel(`%s`) argument index %d \n",kernelName,argCfg);
            printLastError(err);
         }             
      }
   }
   for(argCfg=0; argCfg<openclRuntime.setupArg.argIdx; ++argCfg){
      if(openclRuntime.setupArg.argList[argCfg] != NULL){
         free(openclRuntime.setupArg.argList[argCfg]);
         openclRuntime.setupArg.argList[argCfg] = NULL;
      }
      openclRuntime.setupArg.argIdxes[argCfg] = 0;
      openclRuntime.setupArg.argSize[argCfg] = 0;
   }
   openclRuntime.setupArg.argIdx = 0;
   err = clEnqueueNDRangeKernel(
              openclRuntime.ompclCommandQueue,
              kernel,
              openclRuntime.workdim,0,openclRuntime.globaldim,openclRuntime.localdim,
              0,0,0);
   if(err != 0){
       fprintf(stderr,"Error while launch kernel `%s`\n",kernelName);
       printLastError(err);
   }
   if(!vector_empty(vecReleaseList))
   {
      vector_foreach(vecReleaseList,vec_release_list_releaseMemObj);
      vector_delete(vecReleaseList);
   }
}
void openclLaunch(const char* kernelName){
   int err;
   cl_kernel kernel;
   if(!openclCheckInited()) return ; 
   kernel = clCreateKernel(openclRuntime.ompclProgram,kernelName,&err);
   if(err != 0){
      fprintf(stderr,"Error while create kernel %s\n",kernelName);
      return;
   }
   openclLaunchKernelObject(kernel,kernelName);
   clReleaseKernel(kernel);

}

int  openclShiftPointer(void** ptr,const void* srcPtr, size_t offset){
   cl_buffer_region subbuf;
   cl_mem_flags flag;
   cl_mem ret;
   int err;
   size_t totalSize;
   if(offset == 0){
      *ptr = (void*)srcPtr;
      return 0;
   }
   err = clGetMemObjectInfo((cl_mem)srcPtr,CL_MEM_SIZE,sizeof(totalSize),&totalSize,0);
   if(err != 0){
      fprintf(stderr,"Error while get memory size from Object %p(arg index:2) in openclShiftPointer\n",srcPtr);
      printLastError(err);
      return err;
   }
   subbuf.origin = offset;
   subbuf.size = totalSize - offset;
   err = clGetMemObjectInfo((cl_mem)srcPtr,CL_MEM_FLAGS,sizeof(flag),&flag,0);
   if(err != 0){
      fprintf(stderr,"Error while get flags from Object %p(arg index:2) in openclShiftPointer\n",srcPtr);
      printLastError(err);
      return err;
   }
   ret = clCreateSubBuffer((cl_mem)srcPtr,flag,CL_BUFFER_CREATE_TYPE_REGION,&subbuf,&err);
   if(err != 0){
      fprintf(stderr,"Error while create sub buffer from Object %p(arg index:2) in openclShiftPointer\n",srcPtr);
      printLastError(err);
      return err;
   }
   *ptr = (void*)ret;
   return 0;
}

typedef void* pointer;
void openclLaunchGrid(const char* kernelName,size_t localdim[3],size_t globaldim[3],...){
   va_list va;
   cl_uint argCnt;   
   int err;
   cl_kernel kernel;
   pointer arg;
   int i;
   if(!openclCheckInited()) return ; 
   kernel = clCreateKernel(openclRuntime.ompclProgram,kernelName,&err);
   if(err != 0){
      fprintf(stderr,"Error while create kernel %s\n",kernelName);
      return;
   }
   err = clGetKernelInfo(kernel,CL_KERNEL_NUM_ARGS,sizeof(argCnt),&argCnt,0);
   if(err != 0){
      fprintf(stderr,"Error while get kernel arg info (kernel args) in openclLaunchGrid\n");
      return;
   }
   va_start(va,globaldim);
   for(i=0; i<argCnt; ++i){
      arg = va_arg(va,pointer);
      openclSetArgument(&arg,sizeof(pointer),i);
   }
   va_end(va);
   openclConfigureCall(localdim,globaldim);
   openclLaunchKernelObject(kernel,kernelName);
   clReleaseKernel(kernel);
}

