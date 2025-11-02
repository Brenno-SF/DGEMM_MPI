#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <mpi.h>

// Comandos para compilar e executar:
//mpicc -O3 -march=native -lm -o dgemm_mpi dgemm_mpi.c
//mpirun -np 4 ./dgemm_mpi 2048 

#define blockSize 64  

void randomMatrix(double *aa, int nn){
    int i;
    for (i = 0; i < nn; i++)
        aa[i] = (double)(rand()) / RAND_MAX;
}

double calculateMaxDiff(double *seq, double *par, int z) {
    double maxDiff = 0.0;
    double eps = 1e-9;

    for (int i = 0; i < z; i++) {
        double rel = fabs(seq[i] - par[i]) / (fabs(seq[i]) + eps);
        if (rel > maxDiff) {
            maxDiff = rel;
        }
    }
    return maxDiff;
}

void transpose(int n, double *restrict matrix, double *restrict result) {
    int i, j, ii, jj;
    //block tiling to improve cache performance
    for (ii = 0; ii < n; ii+=blockSize){
        for(jj =0; jj < n; jj+=blockSize){

            for (i = ii; i < ii + blockSize && i < n ; i++) {
                for (j = jj; j < jj + blockSize && j < n; j++) {
                    result[j * n + i] = matrix[i * n + j];
                }

            }
        }
    }
}

void dgemm(int n, int localRows, double alpha, double *restrict a ,double *restrict bT, double beta, double *restrict c) {

    int i, j, k, ii, jj, kk;

    for (ii = 0; ii < localRows; ii += blockSize) {
        for (jj = 0; jj < n; jj += blockSize) {
            for (kk = 0; kk < n; kk += blockSize) {
                for (i = ii; i < ii + blockSize && i < localRows; i += 2) {
                    for (j = jj; j < jj + blockSize && j < n; j += 2) {
                        double c00 = 0.0, c01 = 0.0, c10 = 0.0, c11 = 0.0;
                        for (k = kk; k < kk + blockSize && k < n; k++) {
                            double a0k = a[i * n + k]; 
                            double a1k = (i + 1 < localRows) ? a[(i + 1) * n + k] : 0.0;
                            double b0k = bT[j * n + k];
                            double b1k = (j + 1 < n) ? bT[(j + 1) * n + k] : 0.0;

                            c00 += a0k * b0k;
                            c01 += a0k * b1k;
                            c10 += a1k * b0k;
                            c11 += a1k * b1k;
                        }

                        c[i * n + j] += alpha * c00;
                        if (j + 1 < n) c[i * n + j + 1] += alpha * c01;
                        if (i + 1 < localRows) c[(i + 1) * n + j] += alpha * c10;
                        if (i + 1 < localRows && j + 1 < n) c[(i + 1) * n + j + 1] += alpha * c11;
                    }
                }
            }
        }
    }
}

double runDgemmMpi(int n, double alpha, double beta, int rank, int size, double *restrict a, double *restrict b, double *restrict c) {
    if (n % size != 0) {
        if (rank == 0) 
        fprintf(stderr, "Error: n (%d) must be divisible by number of processes (%d)\n", n, size);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int localN = n / size;

    // Aloca matrizes locais (verifica NULL)
    double *ALocal  = calloc(localN * n, sizeof(double));
    double *BLocal  = calloc(localN * n, sizeof(double));
    double *BTFull  = calloc(n * n, sizeof(double));
    double *CLocal  = calloc(localN * n, sizeof(double));
    double *A = NULL, *B = NULL, *C = NULL;
    double *BFull = calloc(n * n, sizeof(double)); // buffer temporário para montar B completo

    if (!ALocal || !BLocal || !BTFull || !CLocal || !BFull) {
        fprintf(stderr, "[rank %d] Error in memory allocation\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (rank == 0) {
        A = a;
        B = b;
        C = c;

        if (!A || !B || !C) {
            fprintf(stderr, "[rank 0] Error in input matrices\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // Broadcast de parâmetros
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&alpha, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&beta, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    

    // Scatter de A e B (blocos de linhas)
    MPI_Scatter(A, localN * n, MPI_DOUBLE, ALocal, localN * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(B, localN * n, MPI_DOUBLE, BLocal, localN * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Allgather para formar BFull em todos os processos
    MPI_Allgather(BLocal, localN * n, MPI_DOUBLE,
                  BFull, localN * n, MPI_DOUBLE, MPI_COMM_WORLD);

    // Agora BFull contém a matriz B completa em cada processo. Transpõe para BTFull
    transpose(n, BFull, BTFull);

    // Cálculo local DGEMM: ALocal (localN x n) * BTFull (n x n) -> CLocal (localN x n)
    dgemm(n, localN, alpha, ALocal, BTFull, beta, CLocal);

    // Gather do resultado
    MPI_Gather(CLocal, localN * n, MPI_DOUBLE, C, localN * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    
    // libera memória
    free(ALocal); 
    free(BLocal);
    free(BTFull); 
    free(CLocal);
    free(BFull);

    
    return t1 - t0;
}

int main(int argc, char **argv) {
    int n;
    if (argc > 1) 
        n = atoi(argv[1]);

    double alpha = 1.0, beta = 0.0; //valores arbitrários
    int rank, size;
    double *a, *b, *cSeq, *bT, *cMpi;

    int z = n * n;
    
    a = calloc(n * n, sizeof(double));
    b = calloc(n * n, sizeof(double));
    bT = calloc(n * n, sizeof(double));
    cSeq = calloc(n * n, sizeof(double));
    cMpi = calloc(n * n, sizeof(double));

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    if (rank==0){
        printf("Generate random matrix 1 (%d x %d)\n", n, n);
        randomMatrix(a, z);
        printf("Generate random matrix 2 (%d x %d)\n", n, n);
        randomMatrix(b, z);   
    }

    double timeMpi = runDgemmMpi(n, alpha, beta, rank, size, a, b, cMpi);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        transpose(n, b, bT);
        double t0 = MPI_Wtime();
        dgemm(n, n, alpha, a, bT, beta, cSeq);
        double t1 = MPI_Wtime();
        double timeSeq = t1 - t0;

        printf("\nSequential time: %f s\n", timeSeq);
        printf("dgemmMPI time (%d procs): %f s\n", size, timeMpi);

        double speedup = timeSeq / timeMpi;
        printf("Speedup with %d procs: %f\n", size, speedup);

        double efficiency = speedup / size;
        printf("Efficiency with %d procs: %f\n",size, efficiency);

        double maxDiff = calculateMaxDiff(cSeq, cMpi, z);
        printf("Max relative difference between sequential and MPI: %e\n", maxDiff);
    }
    
    free(a); 
    free(b); 
    free(bT); 
    free(cSeq); 
    free(cMpi);

    MPI_Finalize();
    return 0;
}