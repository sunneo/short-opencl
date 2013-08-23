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
static cl_program OMPCLCompileProgram(const char* srcCode,cl_device_id* devid,cl_context ctx)
{
      cl_program retprogram;
            {
             char* prog = (char*)srcCode;
             size_t len = (size_t)strlen(prog);
             const char buildOptions[] = "-w";
             int errret = 0;
             retprogram = clCreateProgramWithSource(ctx,1,(const char**)&prog,&len,&errret);
             if(errret != CL_SUCCESS){
                fprintf(stderr, "Error in OMPCLCompileProgram(clCreateProgramWithSource)\n");
                printLastError(errret);
                exit(0);
             }
             errret = clBuildProgram(retprogram,1,devid,buildOptions,NULL,NULL);
             if (errret != CL_SUCCESS) {
                int status;
                char* programLog;
                size_t logSize=4096;
                cl_device_id device = devid[0];
                // check build error and build status first
                clGetProgramBuildInfo(retprogram, device, CL_PROGRAM_BUILD_STATUS,sizeof(cl_build_status), &status, NULL);
                clGetProgramBuildInfo(retprogram, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
                programLog = (char*) calloc (logSize+1, sizeof(char));
                clGetProgramBuildInfo(retprogram, device, CL_PROGRAM_BUILD_LOG, logSize+1, programLog, NULL);
                printf("Build failed; error=%d, status=%d, programLog:\n\n%s", errret, status, programLog);
                free(programLog);
                exit(0);
             }
         }
   return retprogram;

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
      *program = OMPCLCompileProgram(srcCode,devid,*ctx);
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
static OpenCLRuntimeAPI* openclRuntimeCurrent = &openclRuntime;
 
static void openclRuntimeReleaseOnExit(){
   if(openclRuntimeCurrent->inited){
      clReleaseCommandQueue(openclRuntimeCurrent->ompclCommandQueue);
      clReleaseContext(openclRuntimeCurrent->ompclContext);
      clReleaseProgram(openclRuntimeCurrent->ompclProgram);
      //clUnloadCompiler();
      openclRuntimeCurrent->inited = 0;
   }
}


void openclInitFromSource(const char* src){
   openclInitFromSource2((openclCtx)openclRuntimeCurrent,src);
}
void openclInitFromSource2(openclCtx openclctx,const char* src){
   OMPCLInit(CL_DEVICE_TYPE_GPU,
      &((OpenCLRuntimeAPI*)openclctx)->ompclPlatformID,
      &((OpenCLRuntimeAPI*)openclctx)->ompclDeviceID,
      &((OpenCLRuntimeAPI*)openclctx)->ompclContext, 
      &((OpenCLRuntimeAPI*)openclctx)->ompclCommandQueue,
      src, 
      &((OpenCLRuntimeAPI*)openclctx)->ompclProgram, 
      &((OpenCLRuntimeAPI*)openclctx)->ompclCompiled 
   );
   atexit(openclRuntimeReleaseOnExit);
   ((OpenCLRuntimeAPI*)openclctx)->inited = 1;
}

void openclInitFromFile(const char* file){
   openclInitFromFile2((openclCtx)openclRuntimeCurrent,file);
}

void openclInitFromFile2(openclCtx openclctx,const char* file){
   struct stat s;
   cl_program ret = 0;
   if(((OpenCLRuntimeAPI*)openclctx)->inited){
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
         if(!((OpenCLRuntimeAPI*)openclctx)->inited)
         {
            ((OpenCLRuntimeAPI*)openclctx)->inited=1;
            openclInitFromSource2(openclctx,srcCode);
         }
      }
   }
}

static int openclCheckInited(openclCtx openclctx){
   if(!((OpenCLRuntimeAPI*)openclctx)->inited){
      fprintf(stderr,"OpenCL not inited\n");
      return 0;
   }
   return 1;
}
int openclMalloc(void** ptr,size_t size){
   return openclMalloc2((openclCtx)openclRuntimeCurrent,ptr,size);
}
int openclMalloc2(openclCtx openclctx,void** ptr,size_t size){
    cl_mem ret;
    int err;
    if(!openclCheckInited(openclctx)) return -1;
    ret = clCreateBuffer(((OpenCLRuntimeAPI*)openclctx)->ompclContext, CL_MEM_READ_WRITE, size, NULL,&err);
    vector_push_back(mem_obj_list_get(),mem_obj_new((void*)ret,size));
    *ptr = (void*)ret;
    return err;
}
int openclMemcpy(void* dst, const void* src, size_t size, openclMemcpyKind kind){
   return openclMemcpy2((openclCtx)openclRuntimeCurrent,dst,src,size,kind);
}
int openclMemcpy2(openclCtx openclctx,void* dst, const void* src, size_t size, openclMemcpyKind kind){
   if(!openclCheckInited(openclctx)) return -1;
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
          mem_obj_CL_INVALID_MEM_OBJECT_handler_with_orig_and_offset(CL_INVALID_MEM_OBJECT,dst,&dstPtr,&offset);
          err = clEnqueueWriteBuffer(((OpenCLRuntimeAPI*)openclctx)->ompclCommandQueue,(cl_mem)dst,CL_TRUE,0,size,src,0,0,0);

          return err;
      }
      case openclMemcpyDeviceToHost:
      {
          cl_int err;
          void* srcPtr;
          size_t offset;
          mem_obj_CL_INVALID_MEM_OBJECT_handler_with_orig_and_offset(CL_INVALID_MEM_OBJECT,src,&srcPtr,&offset);
          err = clEnqueueReadBuffer(((OpenCLRuntimeAPI*)openclctx)->ompclCommandQueue,(cl_mem)src,CL_TRUE,0,size,dst,0,0,0);
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
          mem_obj_CL_INVALID_MEM_OBJECT_handler_with_orig_and_offset(CL_INVALID_MEM_OBJECT,src,&srcPtr,&srcOffset);
          mem_obj_CL_INVALID_MEM_OBJECT_handler_with_orig_and_offset(CL_INVALID_MEM_OBJECT,dst,&dstPtr,&dstOffset);
          err = clEnqueueCopyBuffer(((OpenCLRuntimeAPI*)openclctx)->ompclCommandQueue,(cl_mem)src,(cl_mem)dst,0,0,size,0,0,0);
          return err;
      }
   }

}
int openclFree(void* ptr){
   return openclFree2((openclCtx)openclRuntimeCurrent,ptr);
}

int openclFree2(openclCtx openclctx,void* ptr){
   if(!openclCheckInited(openclctx)) return -1;
   return clReleaseMemObject((cl_mem)ptr);
}
int openclThreadSynchronize(){
   return openclThreadSynchronize2((openclCtx)openclRuntimeCurrent);
}
int openclThreadSynchronize2(openclCtx openclctx){
   if(!openclCheckInited(openclctx)) return -1;
   return clFinish(((OpenCLRuntimeAPI*)openclctx)->ompclCommandQueue);
}
void openclSetArgument(void* arg, size_t size, size_t index){
   return openclSetArgument2((openclCtx)openclRuntimeCurrent,arg,size,index);
}
void openclSetArgument2(openclCtx openclctx,void* arg, size_t size, size_t index){
   if(!openclCheckInited(openclctx)) return ;
   if(((OpenCLRuntimeAPI*)openclctx)->setupArg.argIdx >= 256){
      fprintf(stderr,"Count of Argument exceeds 256\n");
      exit(0);
   }
   ((OpenCLRuntimeAPI*)openclctx)->setupArg.argList[((OpenCLRuntimeAPI*)openclctx)->setupArg.argIdx]=(void*)malloc(size);
   memcpy(((OpenCLRuntimeAPI*)openclctx)->setupArg.argList[((OpenCLRuntimeAPI*)openclctx)->setupArg.argIdx],arg,size);
   ((OpenCLRuntimeAPI*)openclctx)->setupArg.argSize[((OpenCLRuntimeAPI*)openclctx)->setupArg.argIdx]=size;
   ((OpenCLRuntimeAPI*)openclctx)->setupArg.argIdxes[((OpenCLRuntimeAPI*)openclctx)->setupArg.argIdx]=index;
   ++((OpenCLRuntimeAPI*)openclctx)->setupArg.argIdx;
}
int openclConfigureCall(size_t localdim[3],size_t globaldim[3]){
   return openclConfigureCall2((openclCtx)openclRuntimeCurrent,localdim,globaldim);
}
int openclConfigureCall2(openclCtx openclctx,size_t localdim[3],size_t globaldim[3]){
   size_t workdim = 3;
   int idx;
   if(!openclCheckInited(openclctx)) return -1; 
   for(idx = 2; idx >0; --idx){
     if(localdim[idx] <= 1 || globaldim[idx]  <= 1)
       --workdim;
   }
   for(idx=0; idx<3; ++idx){
      ((OpenCLRuntimeAPI*)openclctx)->localdim[idx] = localdim[idx];
      ((OpenCLRuntimeAPI*)openclctx)->globaldim[idx] = globaldim[idx];
   }
   ((OpenCLRuntimeAPI*)openclctx)->workdim = workdim;
   if(workdim == 0){
      return -1;
   }
   return 0;
}

static void vec_release_list_releaseMemObj(void* o){
   clReleaseMemObject((cl_mem)o);
}


static void openclLaunchKernelObject2(openclCtx openclctx,cl_kernel kernel,const char* kernelName){
   int argCfg;
   int err;
   Vector* vecReleaseList = vector_create(0);
   for(argCfg=0; argCfg<((OpenCLRuntimeAPI*)openclctx)->setupArg.argIdx; ++argCfg){
      err = clSetKernelArg(
           kernel,
           ((OpenCLRuntimeAPI*)openclctx)->setupArg.argIdxes[argCfg],
           ((OpenCLRuntimeAPI*)openclctx)->setupArg.argSize[argCfg],
           ((OpenCLRuntimeAPI*)openclctx)->setupArg.argList[argCfg]
      );
      /*printf("setKernelArg (%x,idx=%d,size=%d,value=%x)\n",kernel,
          ((OpenCLRuntimeAPI*)openclctx)->setupArg.argIdxes[argCfg],
          ((OpenCLRuntimeAPI*)openclctx)->setupArg.argSize[argCfg],
          *(void**)((OpenCLRuntimeAPI*)openclctx)->setupArg.argList[argCfg]
      );*/
      if(err != 0){
         if(err == CL_INVALID_MEM_OBJECT){
             void* ptr;
             
             mem_obj_CL_INVALID_MEM_OBJECT_handler(
                 err,
                 *((void**)((OpenCLRuntimeAPI*)openclctx)->setupArg.argList[argCfg]),
                 &ptr
             );
             vector_push_back(vecReleaseList,ptr);
             err = clSetKernelArg(
               kernel,
               ((OpenCLRuntimeAPI*)openclctx)->setupArg.argIdxes[argCfg],
               ((OpenCLRuntimeAPI*)openclctx)->setupArg.argSize[argCfg],
               &ptr
             );
         }
         if(err != 0){
            fprintf(stderr,"uncaught exception: Error while setup kernel(`%s`) argument index %d \n",kernelName,argCfg);
            printLastError(err);
         }             
      }
   }
   for(argCfg=0; argCfg<((OpenCLRuntimeAPI*)openclctx)->setupArg.argIdx; ++argCfg){
      if(((OpenCLRuntimeAPI*)openclctx)->setupArg.argList[argCfg] != NULL){
         free(((OpenCLRuntimeAPI*)openclctx)->setupArg.argList[argCfg]);
         ((OpenCLRuntimeAPI*)openclctx)->setupArg.argList[argCfg] = NULL;
      }
      ((OpenCLRuntimeAPI*)openclctx)->setupArg.argIdxes[argCfg] = 0;
      ((OpenCLRuntimeAPI*)openclctx)->setupArg.argSize[argCfg] = 0;
   }
   ((OpenCLRuntimeAPI*)openclctx)->setupArg.argIdx = 0;
   err = clEnqueueNDRangeKernel(
              ((OpenCLRuntimeAPI*)openclctx)->ompclCommandQueue,
              kernel,
              ((OpenCLRuntimeAPI*)openclctx)->workdim,0,((OpenCLRuntimeAPI*)openclctx)->globaldim,((OpenCLRuntimeAPI*)openclctx)->localdim,
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
   openclLaunch2((openclCtx)openclRuntimeCurrent,kernelName);
}
void openclLaunch2(openclCtx openclctx,const char* kernelName){
   int err;
   cl_kernel kernel;
   if(!openclCheckInited(openclctx)) return ; 
   kernel = clCreateKernel(((OpenCLRuntimeAPI*)openclctx)->ompclProgram,kernelName,&err);
   if(err != 0){
      fprintf(stderr,"Error while create kernel %s\n",kernelName);
      return;
   }
   openclLaunchKernelObject2(openclctx,kernel,kernelName);
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
void openclLaunchGrid2(openclCtx openclctx,const char* kernelName,size_t localdim[3],size_t globaldim[3],...){
   va_list va;
   cl_uint argCnt;   
   int err;
   cl_kernel kernel;
   pointer arg;
   int i;
   if(!openclCheckInited(openclctx)) return ; 
   kernel = clCreateKernel(((OpenCLRuntimeAPI*)openclctx)->ompclProgram,kernelName,&err);
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
      openclSetArgument2(openclctx,&arg,sizeof(pointer),i);
   }
   va_end(va);
   openclConfigureCall2(openclctx,localdim,globaldim);
   openclLaunchKernelObject2(openclctx,kernel,kernelName);
   clReleaseKernel(kernel);
}
void openclLaunchGrid(const char* kernelName,size_t localdim[3],size_t globaldim[3],...){
   va_list va;
   cl_uint argCnt;   
   int err;
   cl_kernel kernel;
   pointer arg;
   int i;
   if(!openclCheckInited((openclCtx)openclRuntimeCurrent)) return ; 
   kernel = clCreateKernel(openclRuntimeCurrent->ompclProgram,kernelName,&err);
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
   openclLaunchKernelObject2((openclCtx)openclRuntimeCurrent,kernel,kernelName);
   clReleaseKernel(kernel);
}

openclCtx openclCreateCtx(){
   OpenCLRuntimeAPI* ret = (OpenCLRuntimeAPI*)malloc(sizeof(OpenCLRuntimeAPI));
   memset(ret,0,sizeof(OpenCLRuntimeAPI));
   return (openclCtx)ret;
}


void openclDestroyCtx(openclCtx c){
   OpenCLRuntimeAPI* popenclRuntime;
   popenclRuntime = (OpenCLRuntimeAPI*)c;
   if(popenclRuntime->inited){
      clReleaseCommandQueue(popenclRuntime->ompclCommandQueue);
      clReleaseContext(popenclRuntime->ompclContext);
      clReleaseProgram(popenclRuntime->ompclProgram);
      popenclRuntime->inited = 0;
   }
   free(popenclRuntime);
}

void openclPushCurrent(openclCtx c){
   openclRuntimeCurrent = (OpenCLRuntimeAPI*)c;
}
void openclPopCurrent(openclCtx* c){
   openclPeekCurrent(c);
   openclRuntimeCurrent = &openclRuntime;
}

void openclPeekCurrent(openclCtx* c){
   *c = (openclCtx)openclRuntimeCurrent; 
}

