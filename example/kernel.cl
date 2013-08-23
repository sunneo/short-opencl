
typedef struct StructWithoutPointerButAddress{
   size_t addr0;
   size_t addr1;
}StructWithoutPointerButAddress;

__kernel void kernelFnc(
    int __global* gstore,
    int __global* startptr,
    int __global*  offsetptr,
    int offsetval,
    StructWithoutPointerButAddress __global* strPtr)
{
   unsigned __local store[4];
   if(get_local_id(0) == 0){
      store[0] = (unsigned int)(startptr+offsetval);
      store[1] = (unsigned int)&startptr[offsetval];
      store[2] = (unsigned int)&offsetptr[0];
      store[3] = (unsigned int)offsetptr;
   }
   barrier(CLK_LOCAL_MEM_FENCE);
   if(get_local_id(0) < 4){
//      gstore[get_local_id(0) ] = store[get_local_id(0)];
      ((int __global*)strPtr->addr0)[get_local_id(0) ] = store[get_local_id(0)];
   }
}
