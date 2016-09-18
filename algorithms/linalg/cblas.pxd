cdef extern from "cblas.h" nogil:

	enum CBLAS_ORDER:     CblasRowMajor, CblasColMajor
	enum CBLAS_TRANSPOSE: CblasNoTrans, CblasTrans, CblasConjTrans
	enum CBLAS_UPLO:      CblasUpper, CblasLower
	enum CBLAS_DIAG:      CblasNonUnit, CblasUnit
	enum CBLAS_SIDE:      CblasLeft, CblasRight

	# BLAS level 1 routines
	
	void lib_scopy "cblas_scopy"(const int N, const float *X, const int incX, float *Y, const int incY)
	void lib_dcopy "cblas_dcopy"(const int N, const double *X, const int incX, double *Y, const int incY)
	void lib_ccopy "cblas_ccopy"(const int N, const void *X, const int incX, void *Y, const int incY)
	void lib_zcopy "cblas_zcopy"(const int N, const void *X, const int incX, void *Y, const int incY)
	
	
	void lib_saxpy "cblas_saxpy"(const int N, const float  alpha, const float  *X, const int incX, float  *Y, const int incY )
	void lib_daxpy "cblas_daxpy"(const int N, const double alpha, const double *X, const int incX, double *Y, const int incY )	
	void lib_caxpy "cblas_caxpy"(const int N, const void* alpha, const void *X, const int incX, void *Y, const int incY )
	void lib_zaxpy "cblas_zaxpy"(const int N, const void* alpha, const void *X, const int incX, void *Y, const int incY )

	float lib_snrm2 "cblas_snrm2"(const int N, const float *X, const int incX)
	double lib_dnrm2 "cblas_dnrm2"(const int N, const double *X, const int incX)
	float lib_scnrm2 "cblas_scnrm2"(const int N, const void *X, const int incX)
	double lib_dznrm2 "cblas_dznrm2"(const int N, const void *X,  const int incX)
	
	void lib_sscal "cblas_sscal"(const int N, const float alpha, float *X, const int incX)
	void lib_dscal "cblas_dscal"(const int N, const double alpha, double *X, const int incX)
	void lib_cscal "cblas_cscal"(const int N, const void *alpha, void *X, const int incX)
	void lib_zscal "cblas_zscal"(const int N, const void *alpha, void *X, const int incX)
	void lib_csscal "cblas_csscal"(const int N, const float alpha, void *X, const int incX)
	void lib_zdscal "cblas_zdscal"(const int N, const double alpha, void *X, const int incX)
	
	double lib_ddot "cblas_ddot"( const int N, const double *X,
                      const int incX, const double *Y, const int incY)
	void lib_zdotc_sub "cblas_zdotc_sub"(const int N, const void *X, const int incX, const void *Y, const int incY, void *dotc)
	void lib_zdotu_sub "cblas_zdotu_sub"(const int N, const void *X, const int incX, const void *Y, const int incY, void *dotc)
	# BLAS level 2 routines

	void lib_sgemv "cblas_sgemv"(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA, int M, int N, float  alpha, float  *A, int lda, float  *x, int incX, float  beta, float  *y, int incY)

	void lib_dgemv "cblas_dgemv"(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA, int M, int N, double alpha, double *A, int lda, double *x, int incX, double beta, double *y, int incY)
								 
	void lib_cgemv "cblas_cgemv"(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA, int M, int N, void* alpha, void *A, int lda, void *x, int incX, void* beta, void *y, int incY)

	void lib_zgemv "cblas_zgemv"(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA, int M, int N,void* alpha, void *A, int lda, void *x, int incX, void* beta, void *y, int incY)
								 

	# BLAS level 3 routines
		
	void lib_sgemm "cblas_sgemm"(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, int M, int N, int K, float  alpha, float  *A, int lda, float  *B, int ldb, float  beta, float  *C, int ldc)

	void lib_dgemm "cblas_dgemm"(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, int M, int N, int K, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc)
								 
	void lib_cgemm "cblas_cgemm"(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, int M, int N, int K, void* alpha, void *A, int lda, void *B, int ldb, void* beta, void *C, int ldc)
								 
	void lib_zgemm "cblas_zgemm"(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, int M, int N, int K, void* alpha, void *A, int lda, void *B, int ldb, void* beta, void *C, int ldc)