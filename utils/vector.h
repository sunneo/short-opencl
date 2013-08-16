#ifndef VECTOR_H_
#  define VECTOR_H_

typedef void* VectorIter;
#ifndef __WEAKSYM
#  define __WEAKSYM __attribute((weak))
#endif
typedef struct  Vector
{
    void* a;
    int elesize;
    int back,front,capsize;
    void ( *pop_function ) ( void* );
} Vector;

#  ifdef __cplusplus
extern "C" {
#  endif

    /**
     * vector store data by copy (element size) bytes into pointer.
     * if elesize equal 0, it means the vector stores pointer, instead of
     * storing instance.
     * if data type is not pointer(elesize > 0), user should define push_back, insert
     * wrapper, which prepare data first, and invoke push_back/insert by pass the
     * address of the object.
     *
     * you could use setPopFunction() setup the pop function.
     * thus the vector could be a vector contain objects which maintain the
     * destructor automatically.
     *
     * ex:
     *   void vector_push_back_int(Vector* vector,int i){
     *      vector_push_back(vector,&i);
     *   }
     *   void vector_insert_int(Vector* vector,int idx,int val){
     *      vector_insert(vector,idx,&val);
     *   }
     *   int  vector_intVal(Vector* vector,int idx){
     *      return vector_at(vector,idx)
     *   }
     *   void test(){
     *      Vector* intVec = vector_create(sizeof(int));
     *      vector_push_back_int(intVec,rand());
     *      vector_insert_int(intVec,1);
     *      printf("%d ",vector_intVal(intVec,0));
     *      vector_delete(intVec);
     *   }
     *
     */
     Vector*  vector_create ( int elesize );

    /**
     * the popFunction passed in should be just a finalize function for object.
     * if it is no need to finalize it, just fill 0 to it(default to 0).
     */
     void        vector_setPopFunction ( Vector* a,void ( *popFunction ) ( void* ) )__WEAKSYM ;
     int          vector_empty ( const Vector* a )__WEAKSYM ;
     int          vector_size ( const Vector* a )__WEAKSYM ;
     void*        vector_front ( const Vector* a )__WEAKSYM ;
     void*        vector_back ( const Vector* a )__WEAKSYM ;
     int          vector_backidx ( const Vector* a )__WEAKSYM ;
     int          vector_frontidx ( const Vector* a )__WEAKSYM ;
     void         vector_push_back ( Vector* a,const void* data )__WEAKSYM ;
     void         vector_pop_back ( Vector* a )__WEAKSYM ;
     void         vector_push_front ( Vector* a,const void* data )__WEAKSYM ;
     void         vector_pop_front ( Vector* a )__WEAKSYM ;
     void         vector_swap_s ( Vector* a,int idx1,int idx2 )__WEAKSYM ;
     void         vector_swap ( Vector* a,int idx1,int idx2 )__WEAKSYM ;
     void         vector_insert ( Vector* a,int idx,const void* data )__WEAKSYM ;
     void         vector_erase ( Vector* a,int idx )__WEAKSYM ;
     void         vector_erase_range ( Vector* a,int idxBegin,int idxEnd )__WEAKSYM ;
     void*        vector_at ( const Vector* a,int idx )__WEAKSYM ;
     void         vector_foreach ( Vector* a,void ( *func ) ( void* ele ) )__WEAKSYM ;
     void         vector_clear ( Vector* a )__WEAKSYM ;
     void         vector_delete ( Vector* a )__WEAKSYM ;

#  ifdef __cplusplus
}
#  endif

#endif
