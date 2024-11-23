#include <libgen.h>
#include <limits.h>
#include <math.h>
#include <mpi.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

//  TODO hpc non-blocking send: compute blocks on boundaries first and send, then compute inside, then wait

// TODO could use a subarray type MPI for last submatrices on the right of the global matrix, so no padx (pady does not change perf, only take a little bit of space)
// in fact have to!!!! imagine 9x9 matrix, dims[0]==4 => rn all ranks on the last columns will get 3x12 matrix full of padding (no actual data)

// TODO checkMPISuccess to all MPI calls

#ifdef _OPENMP
#include <omp.h>
#endif

#define unlikely(x) __builtin_expect((x), 0)
#define likely(x) __builtin_expect((x), 1)
#define ABORT() MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE)
#define GET_TIME() (MPI_Wtime())  // wall time

enum neighbor { UP, DOWN, LEFT, RIGHT };

struct parameters_slave {
  double dx, dy, dt, g, gamma;
  int source_type;
};

struct parameters {
  double dx, dy, dt, max_t, g, gamma;
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

static void checkMPISuccess(const int code) {
  if (unlikely(code != MPI_SUCCESS)) {
    char err_str[MPI_MAX_ERROR_STRING];
    int err_len;
    fputs(MPI_Error_string(code, err_str, &err_len) == MPI_SUCCESS ? err_str : "MPI error!", stderr);
    fputc('\n', stderr);
    MPI_Abort(MPI_COMM_WORLD, code);
  }
}

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

// TODO better if pady > padx
// subdomains_dims = {nbr_of_subdomains_on_x, nbr_of_subdomains_on_y}, used to pad data.values (padding is not init'd)
static int read_data(struct data *restrict const data, const char *restrict const filename, const int *restrict const subdomains_dims) {
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
      int mod_tmp;
      const int padx = (mod_tmp = data->nx % subdomains_dims[0]) ? subdomains_dims[0] - mod_tmp : 0;
      const int pady = (mod_tmp = data->ny % subdomains_dims[1]) ? subdomains_dims[1] - mod_tmp : 0;
      data->values = malloc((data->nx + padx) * (data->ny + pady) * sizeof(double));
      if (unlikely(!data->values)) {
        printf("Error: Could not allocate data (%d doubles)\n", N);
        ok = 0;
      } else
        if (padx)
          for (int i = 0; i < data->ny && ok; i++)
            ok = fread(data->values + i * (data->nx + padx), sizeof(double), data->nx, fp) == data->nx;
        else
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

static void interpolate_data(const struct data *const interp, const struct data *const data, const int nx, const int ny, const double dx, const double dy) {
  int i_old = INT_MIN;
  double v00, v01, v10, v11;
  #pragma omp parallel for private(v00, v01, v10, v11) firstprivate(i_old)
  for (int jj = 0; jj < ny; jj++) {
    const double y = jj * dy;
    int j = (int) (y / data->dy), j2;
    if (j >= data->ny - 1) j = j2 = data->ny - 2;
    else j2 = j + 1;

    for (int ii = 0; ii < nx; ii++) {
      const double x = ii * dx;
      int i = (int) (x / data->dx), i2;
      if (i != i_old) {
        if (i >= data->nx - 1) i = i2 = data->nx - 2;
        else i2 = i + 1;

        v00 = GET(data, i, j);
        v10 = GET(data, i2, j);
        v01 = GET(data, i, j2);
        v11 = GET(data, i2, j2);
        i_old = i;
      }
      const double v0 = v00 + ((x - i * data->dx) / data->dx) * (v10 - v00);
      const double v1 = v01 + ((x - i * data->dx) / data->dx) * (v11 - v01);
      SET(interp, ii, jj, v0 + ((y - j * data->dy) / data->dy) * (v1 - v0));
    }
  }
}

int main(int argc, char **argv) {
  checkMPISuccess(MPI_Init(&argc, &argv));

  if (unlikely(argc != 2)) {
    printf("Usage: %s parameter_file\n", argv[0]);
    ABORT();
  }

  int rank, global_size, cart_size, coords[2], neighbors[4], dims[2] = {}, periods[2] = {};
  MPI_Comm cart_comm;
  checkMPISuccess(MPI_Comm_size(MPI_COMM_WORLD, &global_size));
  checkMPISuccess(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  checkMPISuccess(MPI_Dims_create(global_size, 2, dims));
  checkMPISuccess(MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm));
  checkMPISuccess(MPI_Comm_size(cart_comm, &cart_size));
  checkMPISuccess(MPI_Comm_rank(cart_comm, &rank));
  checkMPISuccess(MPI_Cart_coords(cart_comm, rank, 2, coords));
  checkMPISuccess(MPI_Cart_shift(cart_comm, 1, 1, &neighbors[UP], &neighbors[DOWN]));
  checkMPISuccess(MPI_Cart_shift(cart_comm, 0, 1, &neighbors[LEFT], &neighbors[RIGHT]));
  const _Bool has_left_neighbor = neighbors[LEFT] != MPI_PROC_NULL,
              has_right_neighbor = neighbors[RIGHT] != MPI_PROC_NULL,
              has_down_neighbor = neighbors[DOWN] != MPI_PROC_NULL,
              has_up_neighbor = neighbors[UP] != MPI_PROC_NULL;

  struct parameters param;
  struct data h;
  if (!rank) {
    if (unlikely(read_parameters(&param, argv[1]))) ABORT();
    print_parameters(&param);
    if (unlikely(read_data(&h, param.input_h_filename, dims))) ABORT();
  }
  checkMPISuccess(MPI_Bcast(&param, 6, MPI_DOUBLE, 0, MPI_COMM_WORLD));
  checkMPISuccess(MPI_Bcast(&param.source_type, 2, MPI_INT, 0, MPI_COMM_WORLD));
  checkMPISuccess(MPI_Bcast(&h, 2, MPI_INT, 0, MPI_COMM_WORLD));
  checkMPISuccess(MPI_Bcast(&h.dx, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD));

  // gridsize will include padding and be perfectly dividable into dims ranks
  int tmp_mod, *sendcounts, *displs;
  const int gridsizex = (tmp_mod = h.nx % dims[0]) ? h.nx + dims[0] - tmp_mod : h.nx,
            gridsizey = (tmp_mod = h.ny % dims[1]) ? h.ny + dims[1] - tmp_mod : h.ny,
            subsizes[2] = {gridsizex/dims[0], gridsizey/dims[1]};
  MPI_Datatype h_subdomain;
  checkMPISuccess(MPI_Type_vector(subsizes[1], subsizes[0], gridsizex, MPI_DOUBLE, &h_subdomain));
  checkMPISuccess(MPI_Type_create_resized(h_subdomain, 0, sizeof *h.values, &h_subdomain));
  checkMPISuccess(MPI_Type_commit(&h_subdomain));
  if (!rank) {
    if (!(sendcounts = malloc(global_size * 2 * sizeof *sendcounts))) ABORT();
    displs = sendcounts + global_size;
    for (int i=0, disp=0; i<dims[0]; i++, disp+=(subsizes[0]-1)*dims[1]) {
      for (int j=0; j<dims[1]; j++, disp++) {
        const int idx = i * dims[1] + j;
        displs[idx] = disp;
        sendcounts[idx] = 1;
      }
    }
  }

  double *const values = malloc(subsizes[0] * subsizes[1] * sizeof *values);
  // TODO see to do it in place for rank == 0 (https://stackoverflow.com/questions/29415663/how-does-mpi-in-place-work-with-mpi-scatter)
  checkMPISuccess(MPI_Scatterv(h.values, sendcounts, displs, h_subdomain, values, subsizes[0] * subsizes[1], MPI_DOUBLE, 0, MPI_COMM_WORLD));
  h.values = values;
  h.nx = has_right_neighbor || h.nx == dims[0] * subsizes[0] ? subsizes[0] : h.nx % subsizes[0];
  h.ny = has_down_neighbor || h.ny == dims[1] * subsizes[1] ? subsizes[1] : h.ny % subsizes[1];
  
  // infer size of domain from input elevation data
  const double hx = h.nx * h.dx, hy = h.ny * h.dy;
  const int nx = max(1, floor(hx / param.dx));
  const int ny = max(1, floor(hy / param.dy));
  const int nt = floor(param.max_t / param.dt);

  /* Does not work, for the Gatherv
  struct data eta_global = {nx * dims[0], ny * dims[1], param.dx, param.dy};
  if (!rank) eta_global.values = malloc(nx * ny * global_size * sizeof *eta_global.values);
  if (!rank) {
    for (int i=0, disp=0; i<dims[0]; i++, disp+=(nx-1)*dims[1])
      for (int j=0; j<dims[1]; j++, disp++)
        displs[i * dims[1] + j] = disp;
  }
  MPI_Datatype eta_h_subdomain;
  checkMPISuccess(MPI_Type_create_subarray(2, (int[]) {nx * dims[0], ny * dims[1]}, (int[]) {nx, ny}, (int[]) {0, 0}, MPI_ORDER_C, MPI_DOUBLE, &eta_h_subdomain));
  checkMPISuccess(MPI_Type_create_resized(eta_h_subdomain, 0, nx * dims[0] * sizeof *eta.values, &eta_h_subdomain));
  checkMPISuccess(MPI_Type_commit(&eta_h_subdomain)); */

  if (!rank) {
    printf(" - %dx%d divided into %dx%d subdomains each of grid size: %g m x %g m (%d x %d = %d grid points)\n", nx * dims[0], ny * dims[1], dims[0], dims[1], hx, hy, nx, ny, nx * ny);
    printf(" - number of time steps: %d\n", nt);
  }

  struct data eta, u, v;
  init_data(&eta, nx + 1, ny + 1, param.dx, param.dy, 0.);  // GET(&eta, 0, j) and GET(&eta, i, 0) will be the receiving ghost cells; GET(&eta,0,0) will never be used
  init_data(&u, nx + 1, ny, param.dx, param.dy, 0.);  // GET(&u, nx, j) will be the receiving ghost cells
  init_data(&v, nx, ny + 1, param.dx, param.dy, 0.);  // GET(&v, i, ny) will be the receiving ghost cells

  MPI_Datatype vertical_ghost_cells;  // for ghost cells: u_send_to_left, eta_send_to_right
  //MPI_Type_create_subarray(2, (int[]) {nx + 1, ny}, (int[]) {1, ny}, (int[]) {0, 0}, MPI_ORDER_C, MPI_DOUBLE, &vertical_ghost_cells);
  checkMPISuccess(MPI_Type_vector(ny, 1, nx + 1, MPI_DOUBLE, &vertical_ghost_cells));
  checkMPISuccess(MPI_Type_create_resized(vertical_ghost_cells, 0, (nx + 1) * sizeof *eta.values, &vertical_ghost_cells));
  checkMPISuccess(MPI_Type_commit(&vertical_ghost_cells));

  // interpolate bathymetry
  struct data h_interp;
  init_data(&h_interp, nx + 1, ny + 1, param.dx, param.dy, 0.);
  interpolate_data(&h_interp, &h, nx + 1, ny + 1, param.dx, param.dy);
  
  //double *u_send_to_left = has_left_neighbor ? malloc(ny * sizeof *u_send_to_left) : NULL; // TODO check malloc non null
  //double *u_recv_from_right = has_right_neighbor ? malloc(ny * sizeof *u_recv_from_right) : NULL;
  //double *eta_recv_from_left = has_left_neighbor ? malloc(ny * sizeof *eta_recv_from_left) : NULL;
  //double *eta_send_to_right = has_right_neighbor ? malloc(ny * sizeof *eta_send_to_right) : NULL;

  //if (!rank || 1) {
    //long a = rank * (nx + 1) * (ny + 1) + 1;
    //for (int jj = 0; jj <= ny; jj++) {
      //for (int ii = 0; ii <= nx; ii++) {
        //SET(&eta, ii, jj, (double) a);
        ////printf("%d, %d: %g\n", ii, jj, (double) a);
        //a++;
      //}
    //}
    //printf("was set %d, %d: %g\n", 400, 0, GET(&u, 400, 0));
  //}

  const double start = GET_TIME();
  for (int n = 0; n < nt; n++) {

    if (!rank && n && (n % (nt / 10)) == 0) {
      const double time_sofar = GET_TIME() - start;
      const double time_eta = (nt - n) * time_sofar / n;
      printf("Computing step %d/%d (ETA: %g seconds)\n", n, nt, time_eta);
      fflush(stdout);
    }

    // output solution
    if (param.sampling_rate && n && !(n % param.sampling_rate)) {
      struct data blabla;
      init_data(&blabla, nx, ny, param.dx, param.dy, 0.);
      for (int i = 1; i < nx + 1; i++)
        for (int j = 1; j < ny + 1; j++)
          SET(&blabla, i - 1, j - 1, GET(&eta, i, j));
      //MPI_Gatherv(blabla.values, nx * ny, MPI_DOUBLE, eta_global.values, sendcounts, displs, eta_h_subdomain, 0, MPI_COMM_WORLD);

      if (rank) {
        write_data_vtk(&blabla, "water elevation", "example_inputs/simple/output/eta_simple", n);
        //write_data_vtk(&u, "x velocity", param.output_u_filename, n);
        //write_data_vtk(&v, "y velocity", param.output_v_filename, n);
      }
    }
    
    // Prepare boundary data for exchange
    //for (int j = 0; j < ny; j++) {
        ////if (has_left_neighbor) u_send_to_left[j] = GET(&u, 0, j);
        //if (has_right_neighbor) eta_send_to_right[j] = GET(&eta, nx, j + 1);
    //}
    //MPI_Sendrecv(u_send_to_left, has_left_neighbor * ny, MPI_DOUBLE, neighbors[LEFT], 0,
                //u_recv_from_right, has_right_neighbor * ny, MPI_DOUBLE, neighbors[RIGHT], 0, cart_comm, MPI_STATUS_IGNORE);
    checkMPISuccess(MPI_Sendrecv(&GET(&u, 0, 0), has_left_neighbor, vertical_ghost_cells, neighbors[LEFT], 0,
                                 &GET(&u, nx, 0), has_right_neighbor, vertical_ghost_cells, neighbors[RIGHT], 0, cart_comm, MPI_STATUS_IGNORE));
    //MPI_Sendrecv(eta_send_to_right, has_right_neighbor * ny, MPI_DOUBLE, neighbors[RIGHT], 0,
                //eta_recv_from_left, has_left_neighbor * ny, MPI_DOUBLE, neighbors[LEFT], 0, cart_comm, MPI_STATUS_IGNORE);
    checkMPISuccess(MPI_Sendrecv(&GET(&eta, nx, 1), has_right_neighbor, vertical_ghost_cells, neighbors[RIGHT], 0,
                                 &GET(&eta, 0, 1), has_left_neighbor, vertical_ghost_cells, neighbors[LEFT], 0, cart_comm, MPI_STATUS_IGNORE));
    checkMPISuccess(MPI_Sendrecv(&GET(&v, 0, 0), has_down_neighbor * nx, MPI_DOUBLE, neighbors[DOWN], 0,
                                 &GET(&v, 0, ny), has_up_neighbor * nx, MPI_DOUBLE, neighbors[UP], 0, cart_comm, MPI_STATUS_IGNORE));
    checkMPISuccess(MPI_Sendrecv(&GET(&eta, 1, ny + 1), has_up_neighbor * nx, MPI_DOUBLE, neighbors[UP], 0,
                                 &GET(&eta, 1, 0), has_down_neighbor * nx, MPI_DOUBLE, neighbors[DOWN], 0, cart_comm, MPI_STATUS_IGNORE));


    //if (rank) {
      //long a = 1;
      //for (int jj = 0; jj <= ny; jj++) {
        //for (int ii = 0; ii <= nx; ii++, a++) {
          //double b = GET(&eta, ii, jj);
          //if (a != b) printf("%d, %d: %g\n", ii, jj, b);
        //}
      //}
    //}
    //MPI_Barrier(MPI_COMM_WORLD);
    //ABORT();

    //// Place received boundary data into ghost cells
    //for (int j = 0; j < ny; j++) {
        //if (has_left_neighbor) {
          //if (GET(&eta, 0, j + 1) != eta_recv_from_left[j])
            //printf("AZERTYUIOPQSDFGHJKL %g %g\n", GET(&eta, 0, j + 1), eta_recv_from_left[j]);
          //SET(&eta, 0, j + 1, eta_recv_from_left[j]);
        //}
    //}

    // impose boundary conditions
    const double t = n * param.dt;
    if (param.source_type == 1) {
      // sinusoidal velocity on top boundary
      const double A = 5;
      const double f = 1. / 20.;
      for (int j = ny; j--;) {  // CHANGED (question 4)
        if (!has_left_neighbor) SET(&u, 0, j, A * sin(2 * M_PI * f * t));
        if (!has_right_neighbor) SET(&u, nx, j, 0.);
      }
      for (int i = nx; i--;) {
        if (!has_down_neighbor) SET(&v, i, 0, 0.);
        if (!has_up_neighbor) SET(&v, i, ny, 0.);
      }
    } else if (param.source_type == 2) {
      // sinusoidal elevation in the middle of the domain
      const double A = 5;
      const double f = 1. / 20.;
      if (coords[0] == dims[0] / 2 && coords[1] == dims[1] / 2) SET(&eta, nx / 2, ny / 2, A * sin(2 * M_PI * f * t));
    } else {
      // TODO: add other sources
      printf("Error: Unknown source type %d\n", param.source_type);
      ABORT();
    }

    #pragma omp parallel for
    for (int j = 0; j < ny; j++) {  // CHANGED (question 4)
      for (int i = 0; i < nx; i++) {
        // update eta
        const double h_ij = GET(&h_interp, i, j);
        double u_ij = GET(&u, i, j);
        double v_ij = GET(&v, i, j);
        const double eta_ij = GET(&eta, i + 1, j + 1) - param.dt * (
          (GET(&h_interp, i + 1, j) * GET(&u, i + 1, j) - h_ij * u_ij) / param.dx
          + (GET(&h_interp, i, j + 1) * GET(&v, i, j + 1) - h_ij * v_ij) / param.dy);
        SET(&eta, i + 1, j + 1, eta_ij);

        // update u and v
        const double c1 = param.dt * param.g;
        const double c2 = param.dt * param.gamma;
        const double eta_imj = GET(&eta, i, j + 1);
        const double eta_ijm = GET(&eta, i + 1, j);
        u_ij = (1. - c2) * u_ij - c1 / param.dx * (eta_ij - eta_imj);
        v_ij = (1. - c2) * v_ij - c1 / param.dy * (eta_ij - eta_ijm);
        SET(&u, i, j, u_ij);
        SET(&v, i, j, v_ij);
      }
    }
  }
  if (!rank) {
    write_manifest_vtk(param.output_eta_filename, param.dt, nt, param.sampling_rate);
    const double time = GET_TIME() - start;
    printf("Done: %g seconds (%g MUpdates/s)\n", time, 1e-6 * (double)eta.nx * (double)eta.ny * (double)nt / time);
    free(sendcounts);
  }

  checkMPISuccess(MPI_Type_free(&vertical_ghost_cells));
  //checkMPISuccess(MPI_Type_free(&eta_h_subdomain));
  checkMPISuccess(MPI_Type_free(&h_subdomain));
  free_data(&h_interp);
  free_data(&eta);
  free_data(&u);
  free_data(&v);
  checkMPISuccess(MPI_Comm_free(&cart_comm));
  checkMPISuccess(MPI_Finalize());
  return EXIT_SUCCESS;
}

