__kernel void kernelFnc(
    int __global* store,
    int __global* startptr,
    int __global*  offsetptr,
    int offsetval)
{
   if(get_global_id(0) == 0){
      store[0] = (unsigned int)(startptr+offsetval);
      store[1] = (unsigned int)&startptr[offsetval];
      store[2] = (unsigned int)&offsetptr[0];
      store[3] = (unsigned int)offsetptr;
   }
}
