#include "vector.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
__inline static void  vector_empty_rewind ( Vector* a );
__inline static int   vector_idx_out_of_boundary ( const Vector* a, int idx );
__inline static void  vector_resize_handler ( Vector* a );
__inline static int  vector_objSize ( const Vector* a );
__inline static void*  vector_pos_at ( const Vector* a, int idx );
__inline static void*  vector_real_pos ( const Vector* a, int idx );
__inline static void*  vector_virtual_addr ( const Vector* a, int idx );

__inline static void* 
vector_real_pos ( const Vector* a, int idx )
{
   return ( char* ) a->a + ( idx * vector_objSize ( a ) );
}

__inline static void* 
vector_virtual_addr ( const Vector* a, int idx )
{
   return ( char* ) a->a + a->front + ( idx * vector_objSize ( a ) );
}


__inline static void 
vector_write_pointer_at_realpos ( Vector* a, int idx, const void* data )
{
   if ( a->elesize > 0 )
   {
      memcpy ( ( ( char* ) a->a + idx * a->elesize ), data, a->elesize );
      /*((void**)(a->a))[ idx ] = memcpy(malloc(a->elesize),data,a->elesize);*/
   }
   else
   {
      ( ( void** ) ( a->a ) ) [ idx ] = ( void* ) data;
   }
}

__inline static void 
vector_write_pointer_at ( Vector* a, int idx, const void* data )
{
   vector_write_pointer_at_realpos ( a, a->front + idx, data );
}

__inline static void 
vector_delete_at_realpos ( Vector* a, int idx )
{
   if ( a->elesize > 0 )
   {
      a->pop_function ( ( ( char* ) a->a + idx * a->elesize ) );
      /*((void**)(a->a))[ idx ] = memcpy(malloc(a->elesize),data,a->elesize);*/
   }
   else
   {
      a->pop_function ( ( ( void** ) a->a ) [ idx ] );
   }

}

__inline static void 
vector_delete_at ( Vector* a, int idx )
{
   vector_delete_at_realpos ( a, a->front + idx );
}


static void
nofunc ( void* param )
{
   return;
}

Vector* 
vector_create ( int elesize )
{
   Vector* a = ( Vector* ) malloc ( sizeof ( Vector ) );
   a->a = 0;
   a->elesize = elesize;
   a->pop_function = nofunc;
   a->front = a->back = 0;
   a->capsize = 0;
   return a;
}

void 
vector_setPopFunction ( Vector* a, void ( *popFunction ) ( void* ) )
{
   if ( !a ) return;
   a->pop_function = popFunction ? popFunction : nofunc;
}

int 
vector_empty ( const Vector* a )
{
   return a->front == a->back;
}

int 
vector_size ( const Vector* a )
{
   return a->back - a->front;
}

void*  
vector_front ( const Vector* a )
{
   return vector_at ( a, a->front );
}

void* 
vector_back ( const Vector* a )
{
   return vector_at ( a, a->back - 1 ? a->back - 1 : a->back );
}

void 
vector_push_back ( Vector* a, const void* data )
{
   vector_resize_handler ( a );
   vector_write_pointer_at_realpos ( a, a->back, data );
   ++a->back;
}

void 
vector_pop_back ( Vector* a )
{
   if ( vector_empty ( a ) ) return;
   --a->back;
   vector_delete_at_realpos ( a, a->back );
   vector_empty_rewind ( a );
}

__inline static void 
vector_resize_handler ( Vector* a )
{
   if ( a->back + 1 >= a->capsize )
   {
      a->capsize = a->capsize * 2 + 1;
      a->a = realloc ( a->a, a->capsize * vector_objSize ( a ) );
   }
}

void 
vector_push_front ( Vector* a, const void* data )
{
   if ( a->front != 0 )
   {
      --a->front;
      vector_write_pointer_at_realpos ( a, a->front, data );
   }
   else
   {
      ++a->back;
      vector_resize_handler ( a );
      memmove ( vector_virtual_addr ( a, 1 ), ( ( void** ) a->a ), vector_size ( a ) * vector_objSize ( a ) );
      vector_write_pointer_at_realpos ( a, 0, data );
   }
}

__inline static int 
vector_idx_out_of_boundary ( const Vector* a, int idx )
{
   idx += a->front;
   return ( idx ) >= a->back || ( idx < 0 );
}

__inline static void 
vector_empty_rewind ( Vector* a )
{
   if ( vector_empty ( a ) )
   {
      a->front = a->back = 0;
   }
}

void 
vector_pop_front ( Vector* a )
{
   if ( vector_empty ( a ) ) return;
   vector_delete_at ( a, 0 );
   ++a->front;
   vector_empty_rewind ( a );
}

__inline static int 
vector_objSize ( const Vector* a )
{
   if ( a->elesize > 0 )
      return a->elesize;
   else
      return sizeof ( void* );
}


__inline static void* 
vector_pos_at ( const Vector* a, int idx )
{
   if ( a->elesize > 0 )
      return ( ( char* ) a->a ) +  ( a->front + idx ) * vector_objSize ( a );
   else
      return ( ( void** ) ( a->a ) ) [ a->front + idx ];
}



void 
vector_erase ( Vector* a, int idx )
{
   if ( vector_idx_out_of_boundary ( a, idx ) ) return;
   if ( idx == 0 )
   {
      vector_pop_front ( a );
   }
   else
   {
      vector_delete_at ( a, idx );
      memmove ( vector_real_pos ( a, a->front + idx ), vector_real_pos ( a, a->front + idx + 1 ), ( a->back - idx ) * vector_objSize ( a ) );
      vector_pop_back ( a );
   }
}



void 
vector_erase_range ( Vector* a, int idxBegin, int idxEnd )
{
   int i;
   int offset;
   if ( vector_empty ( a ) ) return;
   if ( idxBegin > idxEnd )
   {
      i = idxBegin;
      idxBegin = idxEnd;
      idxEnd = i;
   }
   idxBegin -= ( idxBegin ) * ( idxBegin < 0 );
   idxBegin += a->front;
   idxEnd += a->front;
   idxEnd -= ( idxEnd > a->back ) * ( idxEnd - a->back );
   offset = idxEnd - idxBegin;
   for ( i = idxBegin; i <= idxEnd; ++i )
   {
      vector_delete_at_realpos ( a, i );
   }
   if ( idxBegin == a->front )
   {
      a->front -= offset;
      a->front -= ( a->front ) * ( a->front < 0 );
      idxBegin = 0;
   }
   memmove ( vector_real_pos ( a, idxBegin ), vector_real_pos ( a, idxEnd + 1 ) , ( a->back - idxEnd ) *vector_objSize ( a ) );
   a->back -= ( offset + 1 );
}

void 
vector_insert ( Vector* a, int idx, const void* data )
{
   int size;
   int i;
   if ( idx == 0 )
   {
      vector_push_front ( a, data );
   }
   else
   {
      size = vector_size ( a );
      idx -= ( idx ) * ( idx < 0 );
      idx -= ( idx - size ) * ( idx >= size );
      ++a->back;
      vector_resize_handler ( a );
      memmove ( vector_virtual_addr ( a, idx + 1 ), vector_virtual_addr ( a, idx ), ( a->back - idx ) * vector_objSize ( a ) );
      vector_write_pointer_at_realpos ( a, idx, data );
   }
}

int 
vector_backidx ( const Vector* a )
{
   return vector_size ( a ) - 1;
}

int 
vector_frontidx ( const Vector* a )
{
   return 0;
}


#define defVecSwap(BuiltInType)                          \
   static void                                    \
   vector_swap_##BuiltInType(Vector* a,int idx1, int idx2){ \
      BuiltInType c;                                        \
      idx1+=a->front;                                       \
      idx2+=a->front;                                       \
      c = ((BuiltInType*)a->a)[ idx1 ];                      \
      ((BuiltInType*)a->a)[ idx1 ] = ((BuiltInType*)a->a)[ idx2 ];   \
      ((BuiltInType*)a->a)[ idx2 ] = c;                             \
   }
typedef void* ptr;
typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned long ulong;
defVecSwap ( uchar )
defVecSwap ( ushort )
defVecSwap ( ulong )
defVecSwap ( ptr )
#define CallVecSwap(TYPE,PTR_VEC,IDX1,IDX2) vector_swap_##TYPE(PTR_VEC,IDX1,IDX2)

__inline static void 
swap_mem ( void* a, void* b, int size )
{
   int lsize;
   int csize;
   long _lval;
   char _cval;
   lsize = size / sizeof ( long );
   csize = size % sizeof ( long );
   while ( lsize-- )
   {
      _lval = * ( ( long* ) a );
      * ( ( long* ) a ) = * ( ( long* ) b );
      * ( ( long* ) b ) = _lval;
      a = ( void* ) ( ( char* ) a + sizeof ( long ) );
      b = ( void* ) ( ( char* ) b + sizeof ( long ) );
   }
   while ( csize-- )
   {
      _cval = * ( ( char* ) a );
      * ( ( char* ) a ) = * ( ( char* ) b );
      * ( ( char* ) b ) = _cval;
      a = ( void* ) ( ( char* ) a + sizeof ( char ) );
      b = ( void* ) ( ( char* ) b + sizeof ( char ) );
   }
}

static void 
vector_swap_default ( Vector* a, int idx1, int idx2 )
{
   idx1 += a->front;
   idx2 += a->front;
   swap_mem ( vector_pos_at ( a, idx1 ), vector_pos_at ( a, idx2 ), a->elesize );
}

void 
vector_swap ( Vector* a, int idx1, int idx2 )
{
   switch ( a->elesize )
   {
      case sizeof ( uchar ) :
         CallVecSwap ( uchar, a, idx1, idx2 );
         break;
      case sizeof ( ushort ) :
         CallVecSwap ( ushort, a, idx1, idx2 );
         break;
      case sizeof ( ulong ) :
         CallVecSwap ( ulong, a, idx1, idx2 );
         break;
      default:
         if ( a->elesize <= 0 )
         {
            CallVecSwap ( ptr, a, idx1, idx2 );
         }
         else
         {
            vector_swap_default ( a, idx1, idx2 );
         }
         break;
   }
}

void 
vector_swap_s ( Vector* a, int idx1, int idx2 )
{
   if ( !a ) return;
   idx1 += a->front;
   idx2 += a->front;
   if ( idx1 < 0 ) idx1 = 0;
   if ( idx2 < 0 ) idx2 = 0;
   if ( idx1 >= a->back )  idx1 = a->back - 1;
   if ( idx2 >= a->back )  idx2 = a->back - 1;
   vector_swap ( a, idx1, idx2 );
}


void* 
vector_at ( const Vector* a, int idx )
{
   void* ret;
   if ( vector_idx_out_of_boundary ( a, idx ) ) return 0;
   /*return ((void**)a->a)[ a->front + idx ];*/
   return  vector_pos_at ( a, idx );
}

void 
vector_clear ( Vector* a )
{
   int i;
   if ( a->capsize != 0 )
   {
      vector_foreach ( a, a->pop_function );
      a->front = a->back = 0;
      a->capsize = 0;
      free ( a->a );
      a->a = 0;
   }
}

void 
vector_foreach ( Vector* a, void ( *func ) ( void* ele ) )
{
   int i;
   int size = vector_size ( a );
   for ( i = 0; i < size; ++i )
      func ( vector_at ( a, i ) );
}

void 
vector_delete ( Vector* a )
{
   vector_clear ( a );
   free ( a );
}
