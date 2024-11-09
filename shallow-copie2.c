#include <libgen.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define unlikely(x) __builtin_expect((x), 0)
#define likely(x) __builtin_expect((x), 1)

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef USE_MPI
# include <mpi.h>
# define ABORT() MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE)
# define GET_TIME() (MPI_Wtime())  // wall time
  enum neighbor { UP, DOWN, LEFT, RIGHT };
  struct parameters_slave {
    double dx, dy, dt, g, gamma;
    int source_type;
  };
  static void checkMPISuccess(const int code) {
    if (unlikely(code != MPI_SUCCESS)) {
      char err_str[MPI_MAX_ERROR_STRING];
      int err_len;
      fputs(MPI_Error_string(code, err_str, &err_len) == MPI_SUCCESS ? err_str : "MPI error!", stderr);
      fputc('\n', stderr);
      MPI_Abort(MPI_COMM_WORLD, code);
    }
  }
#else
# define ABORT() exit(EXIT_FAILURE)
# define MPI_PROC_NULL -2
# ifdef _OPENMP
#  define GET_TIME() (omp_get_wtime())  // wall time
# else
#  define GET_TIME() ((double)clock() / CLOCKS_PER_SEC)  // cpu time
# endif
#endif

struct parameters {
  double dx, dy, dt, g, gamma, max_t;
  int source_type, sampling_rate;
  char input_h_filename[256];
  char output_eta_filename[256];
  char output_u_filename[256];
  char output_v_filename[256];
};

struct data {
  int nx, ny;
  double dx, dy;
  double *values;
};

#define GET(data, i, j) ((data)->values[(data)->nx * (j) + (i)])
#define SET(data, i, j, val) ((data)->values[(data)->nx * (j) + (i)] = (val))

static inline int max(const int a, const int b) { return a > b ? a : b;}

static int read_parameters(struct parameters *restrict const param, const char *restrict const filename) {
  FILE *const fp = fopen(filename, "r");
  if (unlikely(!fp)) {
    printf("Error: Could not open parameter file '%s'\n", filename);
    return 1;
  }
  _Bool ok = 1;
  if (likely(ok)) ok = fscanf(fp, "%lf", &param->dx) == 1;
  if (likely(ok)) ok = fscanf(fp, "%lf", &param->dy) == 1;
  if (likely(ok)) ok = fscanf(fp, "%lf", &param->dt) == 1;
  if (likely(ok)) ok = fscanf(fp, "%lf", &param->max_t) == 1;
  if (likely(ok)) ok = fscanf(fp, "%lf", &param->g) == 1;
  if (likely(ok)) ok = fscanf(fp, "%lf", &param->gamma) == 1;
  if (likely(ok)) ok = fscanf(fp, "%d", &param->source_type) == 1;
  if (likely(ok)) ok = fscanf(fp, "%d", &param->sampling_rate) == 1;
  if (likely(ok)) ok = fscanf(fp, "%256s", param->input_h_filename) == 1;
  if (likely(ok)) ok = fscanf(fp, "%256s", param->output_eta_filename) == 1;
  if (likely(ok)) ok = fscanf(fp, "%256s", param->output_u_filename) == 1;
  if (likely(ok)) ok = fscanf(fp, "%256s", param->output_v_filename) == 1;
  fclose(fp);
  if (unlikely(!ok)) {
    printf("Error: Could not read one or more parameters in '%s'\n", filename);
    return 1;
  }
  return 0;
}

static void print_parameters(const struct parameters *const param) {
  printf("Parameters:\n");
  printf(" - grid spacing (dx, dy): %g m, %g m\n", param->dx, param->dy);
  printf(" - time step (dt): %g s\n", param->dt);
  printf(" - maximum time (max_t): %g s\n", param->max_t);
  printf(" - gravitational acceleration (g): %g m/s^2\n", param->g);
  printf(" - dissipation coefficient (gamma): %g 1/s\n", param->gamma);
  printf(" - source type: %d\n", param->source_type);
  printf(" - sampling rate: %d\n", param->sampling_rate);
  printf(" - input bathymetry (h) file: '%s'\n", param->input_h_filename);
  printf(" - output elevation (eta) file: '%s'\n", param->output_eta_filename);
  printf(" - output velocity (u, v) files: '%s', '%s'\n", param->output_u_filename, param->output_v_filename);
}

static int read_data(struct data *restrict const data, const char *restrict const filename) {
  FILE *const fp = fopen(filename, "rb");
  if (unlikely(!fp)) {
    printf("Error: Could not open input data file '%s'\n", filename);
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
    printf("Error reading input data file '%s'\n", filename);
    return 1;
  }
  return 0;
}

static int write_data(const struct data *restrict const data, const char *restrict const filename, const int step) {
  char out[512];
  if (step < 0)
    sprintf(out, "%s.dat", filename);
  else
    sprintf(out, "%s_%d.dat", filename, step);
  FILE *const fp = fopen(out, "wb");
  if (unlikely(!fp)) {
    printf("Error: Could not open output data file '%s'\n", out);
    return 1;
  }
  _Bool ok = 1;
  if (likely(ok)) ok = (fwrite(&data->nx, sizeof(int), 1, fp) == 1);
  if (likely(ok)) ok = (fwrite(&data->ny, sizeof(int), 1, fp) == 1);
  if (likely(ok)) ok = (fwrite(&data->dx, sizeof(double), 1, fp) == 1);
  if (likely(ok)) ok = (fwrite(&data->dy, sizeof(double), 1, fp) == 1);
  const int N = data->nx * data->ny;
  if (likely(ok)) ok = (fwrite(data->values, sizeof(double), N, fp) == N);
  fclose(fp);
  if (unlikely(!ok)) {
    printf("Error writing data file '%s'\n", out);
    return 1;
  }
  return 0;
}

static int write_data_vtk(const struct data *restrict const data, const char *restrict const name, const char *restrict const filename, const int step) {
  char out[512];
  if (step < 0)
    sprintf(out, "%s.vti", filename);
  else
    sprintf(out, "%s_%d.vti", filename, step);

  FILE *const fp = fopen(out, "wb");
  if (unlikely(!fp)) {
    printf("Error: Could not open output VTK file '%s'\n", out);
    return 1;
  }

  const uint64_t num_points = data->nx * data->ny;
  const uint64_t num_bytes = num_points * sizeof(double);

  fprintf(fp, "<?xml version=\"1.0\"?>\n");
  fprintf(fp, "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n");
  fprintf(fp, "  <ImageData WholeExtent=\"0 %d 0 %d 0 0\" Spacing=\"%lf %lf 0.0\">\n", data->nx - 1, data->ny - 1, data->dx, data->dy);
  fprintf(fp, "    <Piece Extent=\"0 %d 0 %d 0 0\">\n", data->nx - 1, data->ny - 1);

  fprintf(fp, "      <PointData Scalars=\"scalar_data\">\n");
  fprintf(fp, "        <DataArray type=\"Float64\" Name=\"%s\" format=\"appended\" offset=\"0\">\n", name);
  fprintf(fp, "        </DataArray>\n");
  fprintf(fp, "      </PointData>\n");

  fprintf(fp, "    </Piece>\n");
  fprintf(fp, "  </ImageData>\n");

  fprintf(fp, "  <AppendedData encoding=\"raw\">\n_");

  fwrite(&num_bytes, sizeof(uint64_t), 1, fp);
  fwrite(data->values, sizeof(double), num_points, fp);

  fprintf(fp, "  </AppendedData>\n");
  fprintf(fp, "</VTKFile>\n");

  fclose(fp);
  return 0;
}

static int write_manifest_vtk(const char *restrict const filename, const double dt, const int nt, const int sampling_rate) {
  char out[512];
  sprintf(out, "%s.pvd", filename);
  const char *const base_filename = basename((char *)filename);

  FILE *const fp = fopen(out, "wb");
  if (unlikely(!fp)) {
    printf("Error: Could not open output VTK manifest file '%s'\n", out);
    return 1;
  }

  fprintf(fp, "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
  fprintf(fp, "  <Collection>\n");
  if (sampling_rate)
    for (int n = 0; n < nt; n++)
      if (!(n % sampling_rate))
        fprintf(fp, "    <DataSet timestep=\"%g\" file='%s_%d.vti'/>\n", n * dt, base_filename, n);

  fprintf(fp, "  </Collection>\n");
  fprintf(fp, "</VTKFile>\n");
  fclose(fp);
  return 0;
}

static int init_data(struct data *const data, const int nx, const int ny, const double dx, const double dy, const double val) {
  data->nx = nx;
  data->ny = ny;
  data->dx = dx;
  data->dy = dy;
  data->values = malloc(nx * ny * sizeof(double));
  if (unlikely(!data->values)){
    printf("Error: Could not allocate data\n");
    return 1;
  }
  for (unsigned i = nx * ny; i--;)
    data->values[i] = val;
  return 0;
}

static void free_data(struct data *const data) {
  free(data->values);
}

static double interpolate_data(const struct data *const data, const double x, const double y) {
  // TODO could store GET values between calls (multiple points could be in the same small square v00 v01 v10 v11)
  int i = (int) (x / data->dx), i2;
  int j = (int) (y / data->dy), j2;
  
  if (i < 0) i = i2 = 0;
  else if (i >= data->nx - 1) i = i2 = data->nx - 2;
  else i2 = i + 1;
  if (j < 0) j = j2 = 0;
  else if (j >= data->ny - 1) j = j2 = data->ny - 2;
  else j2 = j + 1;

  const double v00 = GET(data, i, j);
  const double v10 = GET(data, i2, j);
  const double v01 = GET(data, i, j2);
  const double v11 = GET(data, i2, j2);
  const double v0 = v00 + ((x - i * data->dx) / data->dx) * (v10 - v00);
  const double v1 = v01 + ((x - i * data->dx) / data->dx) * (v11 - v01);
  return v0 + ((y - j * data->dy) / data->dy) * (v1 - v0);
}

int main(int argc, char **argv) {
#ifdef USE_MPI
  checkMPISuccess(MPI_Init(&argc, &argv));
#endif

  if (unlikely(argc != 2)) {
    printf("Usage: %s parameter_file\n", argv[0]);
    ABORT();
  }

  int hor_factor = 1, ver_factor = 1, global_rank = 0, global_size = 1;
  int neighbors[4] = { MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL };
#ifdef USE_MPI
  int cart_rank, cart_size;
  int dims[2] = {};
  int periods[2] = {};
  int coords[2];
  MPI_Comm cart_comm;
  checkMPISuccess(MPI_Comm_size(MPI_COMM_WORLD, &global_size));
  checkMPISuccess(MPI_Comm_rank(MPI_COMM_WORLD, &global_rank));
  checkMPISuccess(MPI_Dims_create(global_size, 2, dims));
  checkMPISuccess(MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm));
  checkMPISuccess(MPI_Comm_size(cart_comm, &cart_size));
  checkMPISuccess(MPI_Comm_rank(cart_comm, &cart_rank));
  checkMPISuccess(MPI_Cart_coords(cart_comm, cart_rank, 2, coords));
  checkMPISuccess(MPI_Cart_shift(cart_comm, 0, 1, &neighbors[UP], &neighbors[DOWN]));
  checkMPISuccess(MPI_Cart_shift(cart_comm, 1, 1, &neighbors[LEFT], &neighbors[RIGHT]));
  hor_factor = dims[0];
  ver_factor = dims[1];
  //if (!cart_rank)  // master thread
#endif
  {
    struct parameters param;
    struct data h, h_new;
    if (!global_rank) {
      if (unlikely(read_parameters(&param, argv[1]))) ABORT();
      print_parameters(&param);
      if (unlikely(read_data(&h, param.input_h_filename))) ABORT();
    } else
      h = (struct data) {};
    MPI_Bcast(&param, 5, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&param.source_type, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&h, 2, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&h.dx, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    puts("HELLO WORLD!1");

    //const int global_nx = h.nx;
    //const int global_ny = h.ny;
    //const int nx = global_nx / hor_factor;
    //const int ny = global_ny / ver_factor;
    //if (!nx || !ny) ABORT();
    
    //printf("Rank = %4d - Coords = (%3d, %3d)"
         //" - Neighbors (up, down, left, right) = (%3d, %3d, %3d, %3d)\n",
            //global_rank, coords[0], coords[1], 
            //neighbors[UP], neighbors[DOWN], neighbors[LEFT], neighbors[RIGHT]);
    
    
    //MPI_Datatype type, resizedtype;
    //int sizes[2]    = {global_nx, global_ny};
    //int subsizes[2] = {nx, ny};
    //int starts[2]   = {0, 0};
    //MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &type);  
    //MPI_Type_create_resized(type, 0, ny * sizeof(double), &resizedtype);
    //MPI_Type_commit(&resizedtype);
    //int *counts, *displs;
    //if (!(counts = malloc(global_size * sizeof *counts)) || !(displs = malloc(global_size * sizeof *displs))) ABORT();
    //memset(counts, 1, global_size);
    //int disp = 0;
    //for (unsigned i = 0; i < dims[0]; i++, disp+=(nx-1)*dims[1]) {
      //for (unsigned j = 0; j < dims[1]; j++, disp++) {
        //displs[i * dims[1] + j] = (_Bool) disp;
        //printf("%d, displs[%d] = %d\n", global_rank, i * dims[1] + j, (_Bool) disp);
      //}
    //}
    //printf("%dx%d %dx%d\n", global_nx, global_ny, nx, ny);
    
    //struct data h_new = {};
    //h_new.values = malloc(nx * ny * sizeof *h_new.values);
    //MPI_Scatterv(h.values, counts, displs, resizedtype, h_new.values, nx*ny, MPI_DOUBLE, 0, MPI_COMM_WORLD);




    const int gridsize=h.nx; // size of grid
    const int gridsizex=h.ny; // size of grid
    const int procgridsize=dims[0];  // size of process grid
    const int procgridsizex=dims[1];  // size of process grid

    //if (rank == 0) {
        ///* fill in the array, and print it */
        //malloc2dchar(&global, gridsize, gridsizex);
        //for (int i=0; i<gridsize; i++) {
            //for (int j=0; j<gridsizex; j++)
                //global[i][j] = (double) '0'+(3*i+j)%10;
        //}


        //printf("Global array is:\n");
        //for (int i=0; i<gridsize; i++) {
            //for (int j=0; j<gridsizex; j++)
                //printf("%lf ", global[i][j]);

            //putchar('\n');
        //}
    //}

    /* create the local array which we'll process */
    //malloc2dchar(&local, gridsize/procgridsize, gridsizex/procgridsizex);
    h_new.values = malloc(gridsize/procgridsize * gridsizex/procgridsizex * sizeof *h_new.values);

    /* create a datatype to describe the subarrays of the global array */
    int sizes[2]    = {gridsize, gridsizex};         /* global size */
    int subsizes[2] = {gridsize/procgridsize, gridsizex/procgridsizex};     /* local size */
    int starts[2]   = {0,0};                        /* where this one starts */
    MPI_Datatype type, subarrtype;
    MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &type);
    MPI_Type_create_resized(type, 0, gridsizex/procgridsizex*sizeof(double), &subarrtype);
    MPI_Type_commit(&subarrtype);


    /* scatter the array to all processors */
    int sendcounts[global_size];
    int displs[global_size];

    if (!global_rank) {
      for (int i=0, disp=0; i<procgridsize; i++, disp+=(gridsize/procgridsize-1)*procgridsizex) {
        for (int j=0; j<procgridsizex; j++, disp++) {
          const int idx = i*procgridsizex+j;
          displs[idx] = disp;
          sendcounts[idx] = 1;
        }
      }
    }


    MPI_Scatterv(global_rank ? NULL : h.values, sendcounts, displs, subarrtype, h_new.values,
                 gridsize*gridsizex/(procgridsize*procgridsizex), MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
                 
                 
                 
    
    int nt = 6, nx = 4, ny = 6;
    
    MPI_Barrier(MPI_COMM_WORLD);
    h_new.nx = gridsize/procgridsize;
    h_new.ny = gridsizex/procgridsizex;
    puts("HELLO WORLD!2");
    for (int i=0; i<gridsize/procgridsize; i++)
        for (int j=0; j<gridsizex/procgridsizex; j++)
          nt += GET(&h_new, i, j);
    ABORT();

    struct data eta, u, v;
    init_data(&eta, nx, ny, param.dx, param.dy, 0.);
    init_data(&u, nx + 1, ny, param.dx, param.dy, 0.);
    init_data(&v, nx, ny + 1, param.dx, param.dy, 0.);
    puts("HELLO WORLD!3");

    // interpolate bathymetry
    struct data h_interp;
    init_data(&h_interp, nx, ny, param.dx, param.dy, 0.);
    puts("HELLO WORLD!3a");

    #pragma omp parallel for collapse(2)
    for (int j = 0; j < ny; j++)  // TODO change from ny to 0
      for (int i = 0; i < nx; i++)
        SET(&h_interp, i, j, interpolate_data(&h, i * param.dx, j * param.dy));

    puts("HELLO WORLD!4");
    MPI_Barrier(MPI_COMM_WORLD);
    double *send_left = neighbors[LEFT] != MPI_PROC_NULL ? malloc(ny * sizeof * send_left) : NULL; // TODO check malloc non null
    double *recv_left = neighbors[LEFT] != MPI_PROC_NULL ? malloc(ny * sizeof * recv_left) : NULL;
    double *send_right = neighbors[RIGHT] != MPI_PROC_NULL ? malloc(ny * sizeof * send_right) : NULL;
    double *recv_right = neighbors[RIGHT] != MPI_PROC_NULL ? malloc(ny * sizeof * recv_right) : NULL;

    const double start = GET_TIME();
    for (int n = 0; n < nt; n++) {

      if (n && (n % (nt / 10)) == 0) {
        const double time_sofar = GET_TIME() - start;
        const double eta = (nt - n) * time_sofar / n;
        printf("Computing step %d/%d (ETA: %g seconds)     \r", n, nt, eta);
        fflush(stdout); // TODO rank 0
      }

      // output solution
      if (param.sampling_rate && !(n % param.sampling_rate)) {
        write_data_vtk(&eta, "water elevation", param.output_eta_filename, n);  // TODO collect
        //write_data_vtk(&u, "x velocity", param.output_u_filename, n);
        //write_data_vtk(&v, "y velocity", param.output_v_filename, n);
      }
      
      // Prepare boundary data for exchange
      for (int j = 0; j < ny; j++) {
          if (send_left) send_left[j] = GET(&eta, 0, j);
          if (send_right) send_right[j] = GET(&eta, nx, j);
      }

      if (send_left) MPI_Sendrecv(send_left, ny, MPI_DOUBLE, neighbors[LEFT], 0, recv_right, ny, MPI_DOUBLE, neighbors[RIGHT], 0, cart_comm, MPI_STATUS_IGNORE);
      if (send_right) MPI_Sendrecv(send_right, ny, MPI_DOUBLE, neighbors[RIGHT], 0, recv_left, ny, MPI_DOUBLE, neighbors[LEFT], 0, cart_comm, MPI_STATUS_IGNORE);
      MPI_Sendrecv(eta.values, nx, MPI_DOUBLE, neighbors[UP], 0, eta.values + nx * ny, nx, MPI_DOUBLE, neighbors[DOWN], 0, cart_comm, MPI_STATUS_IGNORE);
      MPI_Sendrecv(eta.values + nx * ny, nx, MPI_DOUBLE, neighbors[DOWN], 0, eta.values, nx, MPI_DOUBLE, neighbors[UP], 0, cart_comm, MPI_STATUS_IGNORE);

      // Place received boundary data into halos
      for (int j = 0; j < ny; j++) {
          if (send_left) SET(&eta, 0, j, recv_left[j]);
          if (send_right) SET(&eta, nx, j, recv_right[j]);
      }

      // impose boundary conditions
      const double t = n * param.dt;
      if (param.source_type == 1) {
        // sinusoidal velocity on top boundary
        const double A = 5;
        const double f = 1. / 20.;
        for (unsigned i = ny; i--;) {  // CHANGED (question 4)
          if (neighbors[LEFT] != MPI_PROC_NULL) SET(&u, 0, i, 0.);
          if (neighbors[RIGHT] != MPI_PROC_NULL) SET(&u, nx, i, 0.);
        }
        for (unsigned i = nx; i--;) {
          if (neighbors[DOWN] != MPI_PROC_NULL) SET(&v, i, 0, 0.);
          if (neighbors[UP] != MPI_PROC_NULL) SET(&v, i, ny, A * sin(2 * M_PI * f * t));
        }
      } else if (param.source_type == 2) {
        // sinusoidal elevation in the middle of the domain
        const double A = 5;
        const double f = 1. / 20.;
        SET(&eta, nx / 2, ny / 2, A * sin(2 * M_PI * f * t));  // TODO
      } else {
        // TODO: add other sources
        printf("Error: Unknown source type %d\n", param.source_type);
        ABORT();
      }

      #pragma omp parallel for collapse(2)
      for (int j = 0; j < ny; j++) {  // CHANGED (question 4)
        for (int i = 0; i < nx; i++) {
          // update eta
          const double h_ij = GET(&h_interp, i, j);
          double u_ij = GET(&u, i, j);
          double v_ij = GET(&v, i, j);
          const double eta_ij = GET(&eta, i, j) - param.dt * (
            (GET(&h_interp, i >= nx - 1 ? i : i + 1, j) * GET(&u, i + 1, j) - h_ij * u_ij) / param.dx
            + (GET(&h_interp, i, j >= ny - 1 ? j : j + 1) * GET(&v, i, j + 1) - h_ij * v_ij) / param.dy);
          SET(&eta, i, j, eta_ij);

          // update u and v
          const double c1 = param.dt * param.g;
          const double c2 = param.dt * param.gamma;
          const double eta_imj = i ? GET(&eta, i - 1, j) : eta_ij;
          const double eta_ijm = j ? GET(&eta, i, j - 1) : eta_ij;
          u_ij = (1. - c2) * u_ij - c1 / param.dx * (eta_ij - eta_imj);
          v_ij = (1. - c2) * v_ij - c1 / param.dy * (eta_ij - eta_ijm);
          SET(&u, i, j, u_ij);
          SET(&v, i, j, v_ij);
        }
      }
    }

    write_manifest_vtk(param.output_eta_filename, param.dt, nt, param.sampling_rate);

    const double time = GET_TIME() - start;
    printf("\nDone: %g seconds (%g MUpdates/s)\n", time, 1e-6 * (double)eta.nx * (double)eta.ny * (double)nt / time);

    MPI_Type_free(&subarrtype);
    free_data(&h_interp);
    free_data(&eta);
    free_data(&u);
    free_data(&v);
  }
#ifdef USE_MPI
  //else {  // slave threads
    //;
  //}
  checkMPISuccess(MPI_Comm_free(&cart_comm));
  checkMPISuccess(MPI_Finalize());
#endif
  return EXIT_SUCCESS;
}

