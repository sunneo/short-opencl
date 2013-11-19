
__kernel void VectorAdd( float __global* a,  float __global* b,  float __global* c, int iNumElements)
{
    int i;
    int lb = get_global_id(0);
    int ub = iNumElements;
    int step = get_global_size(0);
    i = lb;
    while(i < ub){
       c[ i ] = a[ i ] + b[ i ];
       i+=step;
    }
}
