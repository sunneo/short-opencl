__kernel void kernelfnc(size_t __global* pdata, int size){ 
  int i;
  for(i=get_global_id(0); i<size; i+=get_global_size(0)){
     pdata[ i ] = sizeof(size_t);
  }
}


