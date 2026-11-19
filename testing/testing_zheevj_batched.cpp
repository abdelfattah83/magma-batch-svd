/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ahmad Abdelfattah

       @precisions normal z -> s d c

*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "magma_v2.h"
#include "magma_lapack.h"
#include "magma_operators.h"
#include "testings.h"

#define PRECISION_z
#define COMPLEX

#ifdef MAGMA_HAVE_CUDA
// cusolver
#include <cusolverDn.h>
#ifdef KBLAS_AVAIL
    // kblas
    #include<kblas.h>
#endif
#else
#include<rocblas/rocblas.h>
#include<rocsolver/rocsolver.h>
#endif

/*
rocsolver_zheevj_batched(
    rocblas_handle handle,
    const rocblas_esort esort,
    const rocblas_evect evect,
    const rocblas_fill uplo,
    const rocblas_int n,
    rocblas_double_complex *const A[],
    const rocblas_int lda,
    const double abstol,
    double *residual,
    const rocblas_int max_sweeps,
    rocblas_int *n_sweeps,
    double *W,
    const rocblas_stride strideW,
    rocblas_int *info,
    const rocblas_int batch_count)
*/

#ifdef MAGMA_HAVE_CUDA
    #if defined(PRECISION_z)
        #define magma_zsolver_batch_heevj         cusolverDnZheevjBatched
        #define magma_zsolver_batch_heevj_bufsize cusolverDnZheevjBatched_bufferSize
    #elif defined(PRECISION_c)
        #define magma_csolver_batch_heevj         cusolverDnCheevjBatched
        #define magma_csolver_batch_heevj_bufsize cusolverDnCheevjBatched_bufferSize
    #elif defined(PRECISION_d)
        #define magma_dsolver_batch_heevj         cusolverDnDsyevjBatched
        #define magma_dsolver_batch_heevj_bufsize cusolverDnDsyevjBatched_bufferSize
    #elif defined(PRECISION_s)
        #define magma_ssolver_batch_heevj         cusolverDnSsyevjBatched
        #define magma_ssolver_batch_heevj_bufsize cusolverDnSsyevjBatched_bufferSize
    #endif
#else
    // define which solver to be used
    #define ROCSOLVER_EVJ
    // #define ROCSOLVER_EVD
    #if defined(PRECISION_z)
        #define magma_zsolver_batch_heevd         rocsolver_zheevd_batched
        #define magma_zsolver_batch_heevj         rocsolver_zheevj_batched
        #define magma_zrocblas_complex            rocblas_double_complex
        #define magma_zsolver_batch_heevj_bufsize(...)
    #elif defined(PRECISION_c)
        #define magma_csolver_batch_heevd         rocsolver_cheevd_batched
        #define magma_csolver_batch_heevj         rocsolver_cheevj_batched
        #define magma_crocblas_complex            rocblas_float_complex
        #define magma_csolver_batch_heevj_bufsize(...)
    #elif defined(PRECISION_d)
        #define magma_dsolver_batch_heevd         rocsolver_dsyevd_batched
        #define magma_dsolver_batch_heevj         rocsolver_dsyevj_batched
        #define magma_drocblas_complex            double
        #define magma_dsolver_batch_heevj_bufsize(...)
    #elif defined(PRECISION_s)
        #define magma_ssolver_batch_heevd         rocsolver_ssyevd_batched
        #define magma_ssolver_batch_heevj         rocsolver_ssyevj_batched
        #define magma_srocblas_complex            float
        #define magma_ssolver_batch_heevj_bufsize(...)
    #endif
#endif

#define cond (N == 8 && batchCount == 1)
#define ib   (0)

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zheevd_gpu
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    /* Constants */
    const double d_zero = 0;
    const magma_int_t izero = 0;
    const magma_int_t ione  = 1;

    /* Local variables */
    real_Double_t   gpu_time=0., cpu_time=0.;
    magmaDoubleComplex *h_A, *h_R, *h_work, aux_work[1], unused[1];
    magmaDoubleComplex_ptr d_A;
    #ifdef COMPLEX
    double *rwork, aux_rwork[1];
    double** hrwork_array=NULL;
    magma_int_t lrwork[1] = {-1};
    #endif
    double *dw_magma, *w_magma, *w_lapack, result[4]={0, 0, 0, 0}, eps, runused[1];
    magma_int_t *iwork, aux_iwork[1];
    magma_int_t N, NN, batchCount, info, lwork[1] = {-1}, liwork[1] = {-1}, lda, ldda;
    eps = lapackf77_dlamch( "E" );
    int status = 0;

    magmaDoubleComplex **hA_array = NULL, **dA_array = NULL, **hwork_array = NULL;
    double **hw_array=NULL, **dw_array = NULL;
    magma_int_t *info_array=NULL, *dinfo_array=NULL, **iwork_array;

    magma_opts opts( MagmaOptsBatched );
    opts.parse_opts( argc, argv );
    batchCount = opts.batchcount;

    // checking NoVec requires LAPACK
    opts.lapack |= (opts.check && opts.jobz == MagmaNoVec);

    double tol    = opts.tolerance * lapackf77_dlamch("E");
    double tolulp = opts.tolerance * lapackf77_dlamch("P");

    TESTING_CHECK( magma_malloc(  (void**) &dA_array, batchCount * sizeof(magmaDoubleComplex*) ));
    TESTING_CHECK( magma_malloc(  (void**) &dw_array, batchCount * sizeof(magmaDoubleComplex*) ));

    TESTING_CHECK( magma_malloc_cpu( (void**) &hA_array, batchCount * sizeof(magmaDoubleComplex*) ));
    TESTING_CHECK( magma_malloc_cpu( (void**) &hw_array, batchCount * sizeof(magmaDoubleComplex*) ));

    TESTING_CHECK( magma_malloc_cpu( (void**) &hwork_array, batchCount * sizeof(magmaDoubleComplex*) ));
    TESTING_CHECK( magma_malloc_cpu( (void**) &iwork_array, batchCount * sizeof(magma_int_t*) ));

    TESTING_CHECK( magma_imalloc_cpu( &info_array, batchCount * sizeof(magma_int_t) ));
    TESTING_CHECK( magma_imalloc( &dinfo_array, batchCount * sizeof(magma_int_t) ));

    #ifdef COMPLEX
    TESTING_CHECK( magma_malloc_cpu( (void**) &hrwork_array, batchCount * sizeof(double*) ));
    #endif

    #ifdef MAGMA_HAVE_CUDA
    cusolverDnHandle_t devsolverH = NULL;
    syevjInfo_t heevj_params = NULL;
    const int sort_eig  = 1;
    cusolverEigMode_t devsolver_jobz = (opts.jobz == MagmaNoVec) ? CUSOLVER_EIG_MODE_NOVECTOR : CUSOLVER_EIG_MODE_VECTOR;
    cublasFillMode_t  devsolver_uplo = (opts.uplo == MagmaLower) ? CUBLAS_FILL_MODE_LOWER     : CUBLAS_FILL_MODE_UPPER;

    if(opts.version == 5) {
        status = cusolverDnCreate(&devsolverH);
        assert(CUSOLVER_STATUS_SUCCESS == status);

        /* set cusolver stream */
        status = cusolverDnSetStream(devsolverH, magma_queue_get_cuda_stream( opts.queue ));
        assert(CUSOLVER_STATUS_SUCCESS == status);

        /* step 2: configuration of cusolver heevj */
        status = cusolverDnCreateSyevjInfo(&heevj_params);
        assert(CUSOLVER_STATUS_SUCCESS == status);

        /* default value of tolerance is machine zero */
        status = cusolverDnXsyevjSetTolerance( heevj_params, tol);
        assert(CUSOLVER_STATUS_SUCCESS == status);

        /* default value of max. sweeps is 100 */
        status = cusolverDnXsyevjSetMaxSweeps(heevj_params, 100);
        assert(CUSOLVER_STATUS_SUCCESS == status);

        /* enable sorting */
        status = cusolverDnXsyevjSetSortEig( heevj_params, sort_eig);
        assert(CUSOLVER_STATUS_SUCCESS == status);
    }
    #else
        rocblas_handle devsolverH;
        rocblas_create_handle( &devsolverH );
        rocblas_set_stream(devsolverH, magma_queue_get_hip_stream( opts.queue ));

    #endif


    printf("%% jobz = %s, uplo = %s\n", lapack_vec_const(opts.jobz), lapack_uplo_const(opts.uplo) );

    printf("%% BatchCount     N   CPU Time (ms)   GPU Time (ms)   |S-S_magma|   |A-USU^H|   |I-U^H U|\n");
    printf("%%============================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N  = opts.nsize[itest];
            NN = N * batchCount;
            lda  = N;
            ldda = magma_roundup( N, opts.align );  // multiple of 32 by default

            // query for workspace sizes
            // TODO: move query down to CPU testing when magma routine is available
            lwork[0] = -1;
            lapackf77_zheevd( lapack_vec_const(opts.jobz), lapack_uplo_const(opts.uplo),
                                      &N, NULL, &lda, NULL,
                                      aux_work, lwork,
                                      #ifdef COMPLEX
                                      aux_rwork, lrwork,
                                      #endif
                                      aux_iwork, liwork,
                                      &info );

            lwork[0]  = (magma_int_t) MAGMA_Z_REAL( aux_work[0] );
            #ifdef COMPLEX
            lrwork[0] = (magma_int_t) aux_rwork[0];
            #endif
            liwork[0] = aux_iwork[0];

            /* Allocate host memory for the matrix */
            TESTING_CHECK( magma_zmalloc_cpu( &h_A,      N*lda*batchCount ));
            TESTING_CHECK( magma_dmalloc_cpu( &w_magma,  N*batchCount     ));
            TESTING_CHECK( magma_dmalloc_cpu( &w_lapack, N*batchCount     ));
            #ifdef COMPLEX
            TESTING_CHECK( magma_dmalloc_cpu( &rwork,  lrwork[0]*batchCount ));
            #endif
            TESTING_CHECK( magma_imalloc_cpu( &iwork,  liwork[0]*batchCount ));

            TESTING_CHECK( magma_zmalloc_cpu( &h_R,    N*lda*batchCount    ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_work, lwork[0]*batchCount ));

            TESTING_CHECK( magma_zmalloc( &d_A,      N*ldda*batchCount ));
            TESTING_CHECK( magma_dmalloc( &dw_magma, N*batchCount      ));

            /* Clear eigenvalues, for |S-S_magma| check when fraction < 1. */
            lapackf77_dlaset( "Full", &NN, &ione, &d_zero, &d_zero, w_magma,  &NN );
            lapackf77_dlaset( "Full", &NN, &ione, &d_zero, &d_zero, w_lapack, &NN );

            /* Initialize the matrix */
            for(magma_int_t s = 0; s < batchCount; s++) {
                magma_generate_matrix( opts, N, N, h_A + s*lda*N, lda );
            }
            magma_zsetmatrix( N, NN, h_A, lda, d_A, ldda, opts.queue );

            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            magma_zset_pointer( dA_array, d_A,      ldda, 0, 0, ldda*N, batchCount, opts.queue );
            magma_dset_pointer( dw_array, dw_magma,    N, 0, 0,      N, batchCount, opts.queue );

            if(cond) {
                magma_zprint_gpu(N, N, d_A + ib*ldda*N, ldda, opts.queue);
            }

            if ( opts.version == 1 ) {
                gpu_time = magma_sync_wtime( opts.queue );
                magma_zheevj_batched_small_sm(
                    opts.jobz, opts.uplo, N,
                    dA_array, ldda, dw_array,
                    dinfo_array, batchCount, opts.queue );
                gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;
                magma_dgetvector( NN, dw_magma, 1, w_magma, 1, opts.queue);
            }
            else if (opts.version == 2) {
                gpu_time = magma_sync_wtime( opts.queue );
                magma_zheevj_batched_strided_small_sm(
                    opts.jobz, opts.uplo, N,
                    d_A, ldda, ldda*N,
                    dw_magma, N,
                    dinfo_array, batchCount, opts.queue );
                gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;
                magma_dgetvector( NN, dw_magma, 1, w_magma, 1, opts.queue);
            }
            else if (opts.version == 3) {
                double heevj_tol       = 30;
                magma_int_t sort       = 1;
                magma_int_t ntx        = max(opts.nb, 2);
                magma_int_t max_sweeps = 100;
                gpu_time = magma_sync_wtime( opts.queue );
                magma_zheevj_batched_expert_small_sm(
                    opts.jobz, opts.uplo,
                    N, dA_array, ldda, dw_array, dinfo_array, NULL, NULL,
                    heevj_tol, sort, max_sweeps, ntx, batchCount, opts.queue );
                gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;
                magma_dgetvector( NN, dw_magma, 1, w_magma, 1, opts.queue);
            }
            else if (opts.version == 4) {
                double heevj_tol       = 30;
                magma_int_t sort       = 1;
                magma_int_t ntx        = max(opts.nb, 2);
                magma_int_t max_sweeps = 100;
                gpu_time = magma_sync_wtime( opts.queue );
                magma_zheevj_batched_strided_expert_small_sm(
                    opts.jobz, opts.uplo,
                    N, d_A, ldda, ldda*N, dw_magma, N, dinfo_array, NULL, NULL,
                    heevj_tol, sort, max_sweeps, ntx, batchCount, opts.queue );
                gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;
                magma_dgetvector( NN, dw_magma, 1, w_magma, 1, opts.queue);
            }
            else if (opts.version == 5) {
                // query workspace
                magma_int_t devsolver_lwork[1] = {0};
                magmaDoubleComplex *devsolver_work = NULL;
                magma_zsolver_batch_heevj_bufsize(devsolverH, devsolver_jobz, devsolver_uplo, N, d_A, ldda, dw_magma, devsolver_lwork, heevj_params, batchCount);

                if(devsolver_lwork[0] > 0) {
                    TESTING_CHECK( magma_zmalloc(&devsolver_work, devsolver_lwork[0]) );
                }

                #ifdef MAGMA_HAVE_CUDA
                gpu_time = magma_sync_wtime( opts.queue );
                magma_zsolver_batch_heevj(
                    devsolverH, devsolver_jobz, devsolver_uplo,
                    N, d_A, ldda, dw_magma, devsolver_work,
                    devsolver_lwork[0], dinfo_array, heevj_params, batchCount);
                gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;

                #else

                rocblas_evect devsolver_jobz = (opts.jobz == MagmaNoVec) ? rocblas_evect_none : rocblas_evect_original;
                rocblas_fill  devsolver_uplo = (opts.uplo == MagmaLower) ? rocblas_fill_lower : rocblas_fill_upper;
                #if defined(ROCSOLVER_EVJ) // Jacobi
                double *d_residual = NULL;
                int *d_nsweeps = NULL;
                TESTING_CHECK( magma_dmalloc( &d_residual, batchCount ));
                TESTING_CHECK( magma_imalloc( &d_nsweeps, batchCount ));

                gpu_time = magma_sync_wtime( opts.queue );
                rocsolver_zheevj_batched(
                    devsolverH, rocblas_esort_ascending, devsolver_jobz, devsolver_uplo,
                    N, (magma_zrocblas_complex *const *)dA_array, ldda,
                    0., d_residual, 100, d_nsweeps,
                    dw_magma, N,
                    dinfo_array, batchCount);
                gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;

                magma_free( d_residual );
                magma_free( d_nsweeps );

                #else    // ROCSOLVER_EVD

                double *d_E = NULL;
                TESTING_CHECK( magma_dmalloc( &d_E,N*ldda*batchCount ));

                gpu_time = magma_sync_wtime( opts.queue );
                magma_zsolver_batch_heevj(
                    devsolverH, devsolver_jobz, devsolver_uplo,
                    N, (magma_zrocblas_complex *const *)dA_array, ldda, dw_magma, N,
                    d_E, ldda*N, dinfo_array, batchCount);
                gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;

                magma_free( d_E );
                #endif

                #endif

                // copy results back
                magma_dgetvector( NN, dw_magma, 1, w_magma, 1, opts.queue);

                if(devsolver_work != NULL) {
                    magma_free(devsolver_work);
                }
            }
            else {
                gpu_time = magma_wtime();
                for(magma_int_t s = 0; s < batchCount; s++) {
                    magmaDoubleComplex *hAtmp = NULL;
                    magma_zmalloc_cpu(&hAtmp, lda * N);
                    magma_zgetmatrix(N, N, d_A + s*ldda*N, ldda, hAtmp, lda, opts.queue);

                    lapackf77_zheevd( lapack_vec_const(opts.jobz), lapack_uplo_const(opts.uplo),
                                          &N, hAtmp, &lda, w_magma + s * N,
                                          h_work, lwork,
                                          #ifdef COMPLEX
                                          rwork,  lrwork,
                                          #endif
                                          iwork,  liwork,
                                          &info_array[s] );
                    magma_zsetmatrix(N, N, hAtmp, lda, d_A + s*ldda*N, ldda, opts.queue);
                    magma_dsetvector(N, w_magma + s*N, 1, dw_magma + s*N, 1, opts.queue);
                    magma_isetvector(batchCount, info_array, 1, dinfo_array, 1, opts.queue);
                    magma_free_cpu( hAtmp );
                }
                gpu_time = magma_wtime() - gpu_time;
            }

            if(cond) {
                magma_dprint_gpu(N, 1, dw_magma + ib*N, N, opts.queue);
                magma_zprint_gpu(N, N, d_A + ib*ldda*N, ldda, opts.queue);
            }

            // check info
            magma_igetvector(batchCount, dinfo_array, 1, info_array, 1, opts.queue);
            for(magma_int_t s = 0; s < batchCount; s++) {
                if (info_array[s] != 0) {
                    printf("magma_zheevj_batched returned error %lld for matrix %lld: %s.\n",
                        (long long) info_array[s], (long long) s, magma_strerror( info_array[s] ));
                }
            }

            bool okay = true;
            if ( opts.check && opts.jobz != MagmaNoVec ) {
                /* =====================================================================
                   Check the results following the LAPACK's [zcds]drvst routine.
                   A is factored as A = U S U^H and the following 3 tests computed:
                   (1)    | A - U S U^H | / ( |A| N ) if all eigenvectors were computed
                          | U^H A U - S | / ( |A| N ) otherwise
                   (2)    | I - U^H U   | / ( N )
                   (3)    | S(with U) - S(w/o U) | / | S |    // currently disabled, but compares to LAPACK
                   =================================================================== */
                magma_zgetmatrix( N, NN, d_A, ldda, h_R, lda, opts.queue );

                magmaDoubleComplex *work;
                TESTING_CHECK( magma_zmalloc_cpu( &work, 2*N*N ));

                // e is unused since kband=0; tau is unused since itype=1
                result[0] = 0;
                result[1] = 0;
                for(magma_int_t s = 0; s < batchCount; s++) {
                    double iresult[2] = {0, 0};
                    lapackf77_zhet21( &ione, lapack_uplo_const(opts.uplo), &N, &izero,
                                      h_A + s*lda*N, &lda,
                                      w_magma + s*N, runused,
                                      h_R + s*lda*N, &lda,
                                      h_R + s*lda*N, &lda,
                                      unused, work,
                                      #ifdef COMPLEX
                                      rwork,
                                      #endif
                                      iresult );

                    result[0] = magma_max_nan( result[0], iresult[0] );
                    result[1] = magma_max_nan( result[1], iresult[1] );

                }

                result[0] *= eps;
                result[1] *= eps;

                magma_free_cpu( work );  work=NULL;
            }

            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                // setup pointers
                for(magma_int_t s = 0; s < batchCount; s++) {
                    hA_array[s]    = h_A + s * lda * N;
                    hw_array[s]    = w_lapack + s * N;
                    hwork_array[s] = h_work + s * lwork[0];
                    iwork_array[s] = iwork  + s * liwork[0];
                    #ifdef COMPLEX
                    hrwork_array[s] = rwork + s * lrwork[0];
                    #endif
                }

                #if 0
                cpu_time = magma_wtime();
                for(magma_int_t s = 0; s < batchCount; s++) {
                    lapackf77_zheevd( lapack_vec_const(opts.jobz), lapack_uplo_const(opts.uplo),
                                      &N, h_A + s*lda*N, &lda, w_lapack + s*N,
                                      h_work, lwork,
                                      #ifdef COMPLEX
                                      rwork,  lrwork,
                                      #endif
                                      iwork,  liwork,
                                      &info );

                    if (info != 0) {
                        printf("lapackf77_zheevd returned error %lld for matrix %lld: %s.\n",
                               (long long) info, (long long) s, magma_strerror( info ));
                    }
                }
                cpu_time = magma_wtime() - cpu_time;
                #else
                cpu_time = magma_wtime();
                lapack_zheevd_batched(
                    opts.jobz, opts.uplo,
                    N, hA_array, lda, hw_array,
                    hwork_array, lwork,
                    #ifdef COMPLEX
                    hrwork_array, lrwork,
                    #endif
                    iwork_array, liwork,
                    info_array, batchCount );
                cpu_time = magma_wtime() - cpu_time;

                // check info
                for(magma_int_t s = 0; s < batchCount; s++) {
                    if (info_array[s] != 0) {
                        printf("lapackf77_zheevd returned error %lld for matrix %lld: %s.\n",
                               (long long) info_array[s], (long long) s, magma_strerror( info_array[s] ));
                    }

                }
                #endif

                // compare eigenvalues
                result[3] = 0;
                for(magma_int_t s = 0; s < batchCount; s++) {
					double maxw=0, diff=0;
					for( int j=0; j < N; j++ ) {
						maxw = magma_max_nan(maxw, fabs(w_magma [s*N + j]));
						maxw = magma_max_nan(maxw, fabs(w_lapack[s*N + j]));
						diff = magma_max_nan(diff, fabs(w_magma[s*N + j] - w_lapack[s*N + j]));
					}
					result[3] = magma_max_nan( result[3], diff / (N*maxw) );
                }

                okay = okay && (result[3] < tolulp);
                printf("  %10lld %5lld   %9.4f        %9.4f         %8.2e  ",
                       (long long) batchCount, (long long) N, cpu_time*1000., gpu_time*1000., result[3] );
            }
            else {
                printf("  %10lld %5lld      ---           %9.4f           ---     ",
                       (long long) batchCount, (long long) N, gpu_time*1000.);
            }

            // print error checks
            if ( opts.check && opts.jobz != MagmaNoVec ) {
                okay = okay && (result[0] < tol) && (result[1] < tol);
                printf("    %8.2e    %8.2e", result[0], result[1] );
            }
            else {
                printf("      ---         ---   ");
            }
            printf("   %s\n", (okay ? "ok" : "failed"));
            status += ! okay;

            magma_free_cpu( h_A    );
            magma_free_cpu( w_magma     );
            magma_free_cpu( w_lapack     );
            #ifdef COMPLEX
            magma_free_cpu( rwork  );
            #endif
            magma_free_cpu( iwork  );

            magma_free_cpu( h_R    );
            magma_free_cpu( h_work );

            magma_free( d_A );
            magma_free( dw_magma );

            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    magma_free( dA_array );
    magma_free( dw_array );
    magma_free( dinfo_array );
    magma_free_cpu( hA_array );
    magma_free_cpu( hw_array );
    magma_free_cpu( info_array );
    magma_free_cpu( hwork_array );
    magma_free_cpu( iwork_array );
    #ifdef COMPLEX
    magma_free_cpu( hrwork_array );
    #endif

    if(opts.version == 5) {
        #ifdef MAGMA_HAVE_CUDA
        cusolverDnDestroySyevjInfo(heevj_params);
        cusolverDnDestroy(devsolverH);
        #else
        rocblas_destroy_handle( devsolverH );
        #endif
    }

    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
