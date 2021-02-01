
IF MKL_INTERFACE_LAYER:
    ctypedef long long int MKL_INT
ELSE:
    ctypedef int MKL_INT

cdef extern from "mkl.h" nogil:
    # see $CONDA_PREFIX/include/mkl_cblas.h

    enum CBLAS_TRANSPOSE: CblasNoTrans, CblasTrans, CblasConjTrans
    enum CBLAS_LAYOUT: CblasRowMajor, CblasColMajor

    int mkl_set_interface_layer(int required_interface)

    void cblas_dscal(const MKL_INT N, const double alpha, double *X, const MKL_INT incX);
    void cblas_zscal(const MKL_INT N, const void *alpha, void *X, const MKL_INT incX);
    void cblas_zdscal(const MKL_INT N, const double alpha, void *X, const MKL_INT incX);

    void cblas_daxpy(const MKL_INT N, const double alpha, const double *X, const MKL_INT incX, double *Y, const MKL_INT incY);
    void cblas_zaxpy(const MKL_INT N, const void *alpha, const void *X, const MKL_INT incX, void *Y, const MKL_INT incY);

    double cblas_ddot(const MKL_INT N, const double *X, const MKL_INT incX, const double *Y, const MKL_INT incY);
    void cblas_zdotc_sub(const MKL_INT N, const void *X, const MKL_INT incX, const void *Y, const MKL_INT incY, void *dotc);
    void cblas_zdotu_sub(const MKL_INT N, const void *X, const MKL_INT incX, const void *Y, const MKL_INT incY, void *dotu);

    void cblas_dgemm(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb, const MKL_INT m, const MKL_INT n, const MKL_INT k, const double alpha, const double *a, const MKL_INT lda, const double *b, const MKL_INT ldb, const double beta, double *c, const MKL_INT ldc);
    void cblas_zgemm(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb, const MKL_INT m, const MKL_INT n, const MKL_INT k, const void * alpha, const void *a, const MKL_INT lda, const void *b, const MKL_INT ldb, const void * beta, void *c, const MKL_INT ldc);

    void cblas_dgemm_batch(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE* transa_array, const CBLAS_TRANSPOSE* transb_array, const MKL_INT* m_array, const MKL_INT* n_array, const MKL_INT* k_array, const double* alpha_array, const double **a_array, const MKL_INT* lda_array, const double **b_array, const MKL_INT* ldb_array, const double* beta_array, double **c_array, const MKL_INT* ldc_array, const MKL_INT group_count, const MKL_INT* group_size);
    void cblas_zgemm_batch(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE* transa_array, const CBLAS_TRANSPOSE* transb_array, const MKL_INT* m_array, const MKL_INT* n_array, const MKL_INT* k_array, const void *alpha_array, const void **a_array, const MKL_INT* lda_array, const void **b_array, const MKL_INT* ldb_array, const void *beta_array, void **c_array, const MKL_INT* ldc_array, const MKL_INT group_count, const MKL_INT* group_size);
