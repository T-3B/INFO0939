#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "mpi.h"

#define unlikely(x) __builtin_expect((x), 0)
#define likely(x) __builtin_expect((x), 1)


#define GET(data, i, j) ((data)->values[(data)->nx * (j) + (i)])
#define SET(data, i, j, val) ((data)->values[(data)->nx * (j) + (i)] = (val))

struct data {
  int nx, ny;
  double dx, dy;
  double *values;
};
static int read_data(struct data *restrict const data) {
  FILE *const fp = fopen("h_simple.dat", "rb");
  if (unlikely(!fp)) {
    printf("Error: Could not open input data file \n");
    return 1;
  }
  _Bool ok = 1;
  if (likely(ok)) ok = fread(&data->nx, sizeof(int), 1, fp) == 1;
  if (likely(ok)) ok = fread(&data->ny, sizeof(int), 1, fp) == 1;
  if (likely(ok)) ok = fread(&data->dx, sizeof(double), 1, fp) == 1;
  if (likely(ok)) ok = fread(&data->dy, sizeof(double), 1, fp) == 1;
  if (likely(ok)) {
    const int N = data->nx * data->ny;
    if (unlikely(N <= 0)) {
      printf("Error: Invalid number of data points %d\n", N);
      ok = 0;
    } else {
      data->values = malloc(N * sizeof(double));
      if (unlikely(!data->values)) {
        printf("Error: Could not allocate data (%d doubles)\n", N);
        ok = 0;
      } else
        ok = fread(data->values, sizeof(double), N, fp) == N;
    }
  }
  fclose(fp);
  if (unlikely(!ok)) {
    printf("Error reading input data file\n");
    return 1;
  }
  return 0;
}

int malloc2dchar(double ***array, int n, int m) {

    /* allocate the n*m contiguous items */
    double *p = malloc(n*m*sizeof(double));
    if (!p) return -1;

    /* allocate the row pointers into the memory */
    (*array) = malloc(n*sizeof(double*));
    if (!(*array)) {
       free(p);
       return -1;
    }

    /* set up the pointers into the contiguous memory */
    for (int i=0; i<n; i++)
       (*array)[i] = &(p[i*m]);

    return 0;
}

int free2dchar(double ***array) {
    /* free the memory - the first element of the array is at the start */
    free(&((*array)[0][0]));

    /* free the pointers into the memory */
    free(*array);

    return 0;
}

int main(int argc, char **argv) {
    struct data h;
    read_data(&h);
    double **global, **local;
    const int gridsize=10;//h.nx; // size of grid
    const int gridsizex=8;//h.ny; // size of grid
    const int procgridsize=1;  // size of process grid
    const int procgridsizex=2;  // size of process grid
    int rank, size;        // rank of current process and no. of processes

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    if (size != procgridsize*procgridsizex) {
        fprintf(stderr,"%s: Only works with np=%d for now\n", argv[0], procgridsize * procgridsizex);
        MPI_Abort(MPI_COMM_WORLD,1);
    }


    if (rank == 0) {
        /* fill in the array, and print it */
        malloc2dchar(&global, gridsize, gridsizex);
        for (int i=0; i<gridsize; i++) {
            for (int j=0; j<gridsizex; j++)
                global[i][j] = (double) '0'+(3*i+j)%10;
        }


        printf("Global array is:\n");
        for (int i=0; i<gridsize; i++) {
            for (int j=0; j<gridsizex; j++)
                printf("%lf ", global[i][j]);

            putchar('\n');
        }
    }

    /* create the local array which we'll process */
    malloc2dchar(&local, gridsize/procgridsize, gridsizex/procgridsizex);

    /* create a datatype to describe the subarrays of the global array */

    int sizes[2]    = {gridsize, gridsizex};         /* global size */
    int subsizes[2] = {gridsize/procgridsize, gridsizex/procgridsizex};     /* local size */
    int starts[2]   = {0,0};                        /* where this one starts */
    MPI_Datatype type, subarrtype;
    MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &type);
    MPI_Type_create_resized(type, 0, gridsizex/procgridsizex*sizeof(double), &subarrtype);
    MPI_Type_commit(&subarrtype);

    double *globalptr=NULL;
    if (rank == 0) globalptr = &(global[0][0]);

    /* scatter the array to all processors */
    int sendcounts[procgridsize*procgridsizex];
    int displs[procgridsize*procgridsizex];

    if (rank == 0) {
        for (int i=0; i<procgridsize*procgridsizex; i++) sendcounts[i] = 1;
        int disp = 0;
        for (int i=0; i<procgridsize; i++) {
            for (int j=0; j<procgridsizex; j++) {
                displs[i*procgridsizex+j] = disp;
                disp += 1;
            }
            disp += ((gridsize/procgridsize)-1)*procgridsizex;
        }
    }


    MPI_Scatterv(globalptr, sendcounts, displs, subarrtype, &(local[0][0]),
                 gridsize*gridsizex/(procgridsize*procgridsizex), MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    /* now all processors print their local data: */

    for (int p=0; p<size; p++) {
        if (rank == p) {
            printf("Local process on rank %d is:\n", rank);
            for (int i=0; i<gridsize/procgridsize; i++) {
                putchar('|');
                for (int j=0; j<gridsizex/procgridsizex; j++) {
                    printf("%lf ", local[i][j]);
                }
                puts("|\n");
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    /* now each processor has its local array, and can process it */
    for (int i=0; i<gridsize/procgridsize; i++) {
        for (int j=0; j<gridsizex/procgridsizex; j++) {
            local[i][j] = (double) rank;
        }
    }

    /* it all goes back to process 0 */
    MPI_Gatherv(&(local[0][0]), gridsize*gridsizex/(procgridsize*procgridsizex),  MPI_DOUBLE,
                 globalptr, sendcounts, displs, subarrtype,
                 0, MPI_COMM_WORLD);

    /* don't need the local data anymore */
    free2dchar(&local);

    /* or the MPI data type */
    MPI_Type_free(&subarrtype);

    if (rank == 0) {
        printf("Processed grid:\n");
        for (int i=0; i<gridsize; i++) {
            for (int j=0; j<gridsizex; j++) {
                printf("%lf ", global[i][j]);
            }
            printf("\n");
        }

        free2dchar(&global);
    }


    MPI_Finalize();

    return 0;
}
