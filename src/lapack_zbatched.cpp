/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

       @author Ahmad Abdelfattah

       Implementation of batch LAPACK on the host ( CPU ) using OpenMP
*/
#include "magma_internal.h"

#if defined(_OPENMP)
#include <omp.h>
#include "magma_threadsetting.h"
#endif

#define COMPLEX

/*******************************************************************************/
extern "C" void
lapack_zheevd_batched(
    magma_vec_t jobz, magma_uplo_t uplo,
    magma_int_t n,
    magmaDoubleComplex **hA_array, magma_int_t lda,
    double             **hw_array,
    magmaDoubleComplex **hwork_array, magma_int_t* lwork,
    #ifdef COMPLEX
    double **rwork_array, magma_int_t* lrwork,
    #endif
    magma_int_t **iwork_array, magma_int_t* liwork,
    magma_int_t *info_array, magma_int_t batchCount )
{
    #if defined(_OPENMP)
    magma_int_t nthreads = magma_get_lapack_numthreads();
    magma_set_lapack_numthreads(1);
    magma_set_omp_numthreads(nthreads);
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int i=0; i < batchCount; i++) {
        lapackf77_zheevd( lapack_vec_const(jobz), lapack_uplo_const(uplo),
                          &n, hA_array[i], &lda, hw_array[i],
                          hwork_array[i], lwork,
                          #ifdef COMPLEX
                          rwork_array[i], lrwork,
                          #endif
                          iwork_array[i], liwork,
                          &info_array[i] );
    }
    #if defined(_OPENMP)
    magma_set_lapack_numthreads(nthreads);
    #endif
}

