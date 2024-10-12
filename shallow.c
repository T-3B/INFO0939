#include <libgen.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef MPI_VERSION
#include <mpi.h>
#define GET_TIME() (MPI_Wtime())  // wall time
#elifdef _OPENMP
#define GET_TIME() (omp_get_wtime())  // wall time
#else
#define GET_TIME() ((double)clock() / CLOCKS_PER_SEC)  // cpu time
#endif

struct parameters {
  double dx, dy, dt, max_t;
  double g, gamma;
  int source_type;
  int sampling_rate;
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
  if (!fp) {
    printf("Error: Could not open parameter file '%s'\n", filename);
    return 1;
  }
  int ok = 1;
  if (ok) ok = (fscanf(fp, "%lf", &param->dx) == 1);
  if (ok) ok = (fscanf(fp, "%lf", &param->dy) == 1);
  if (ok) ok = (fscanf(fp, "%lf", &param->dt) == 1);
  if (ok) ok = (fscanf(fp, "%lf", &param->max_t) == 1);
  if (ok) ok = (fscanf(fp, "%lf", &param->g) == 1);
  if (ok) ok = (fscanf(fp, "%lf", &param->gamma) == 1);
  if (ok) ok = (fscanf(fp, "%d", &param->source_type) == 1);
  if (ok) ok = (fscanf(fp, "%d", &param->sampling_rate) == 1);
  if (ok) ok = (fscanf(fp, "%256s", param->input_h_filename) == 1);
  if (ok) ok = (fscanf(fp, "%256s", param->output_eta_filename) == 1);
  if (ok) ok = (fscanf(fp, "%256s", param->output_u_filename) == 1);
  if (ok) ok = (fscanf(fp, "%256s", param->output_v_filename) == 1);
  fclose(fp);
  if (!ok) {
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
  if (!fp) {
    printf("Error: Could not open input data file '%s'\n", filename);
    return 1;
  }
  int ok = 1;
  if (ok) ok = (fread(&data->nx, sizeof(int), 1, fp) == 1);
  if (ok) ok = (fread(&data->ny, sizeof(int), 1, fp) == 1);
  if (ok) ok = (fread(&data->dx, sizeof(double), 1, fp) == 1);
  if (ok) ok = (fread(&data->dy, sizeof(double), 1, fp) == 1);
  if (ok) {
    const int N = data->nx * data->ny;
    if (N <= 0) {
      printf("Error: Invalid number of data points %d\n", N);
      ok = 0;
    } else {
      data->values = malloc(N * sizeof(double));
      if (!data->values) {
        printf("Error: Could not allocate data (%d doubles)\n", N);
        ok = 0;
      } else
        ok = (fread(data->values, sizeof(double), N, fp) == N);
    }
  }
  fclose(fp);
  if (!ok) {
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
  if (!fp) {
    printf("Error: Could not open output data file '%s'\n", out);
    return 1;
  }
  int ok = 1;
  if (ok) ok = (fwrite(&data->nx, sizeof(int), 1, fp) == 1);
  if (ok) ok = (fwrite(&data->ny, sizeof(int), 1, fp) == 1);
  if (ok) ok = (fwrite(&data->dx, sizeof(double), 1, fp) == 1);
  if (ok) ok = (fwrite(&data->dy, sizeof(double), 1, fp) == 1);
  const int N = data->nx * data->ny;
  if (ok) ok = (fwrite(data->values, sizeof(double), N, fp) == N);
  fclose(fp);
  if (!ok) {
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
  if (!fp) {
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
  if (!fp) {
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
  if (!data->values){
    printf("Error: Could not allocate data\n");
    return 1;
  }
  for (int i = 0; i < nx * ny; i++)
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

int main(const int argc, const char *const *const argv) {
  if (argc != 2) {
    printf("Usage: %s parameter_file\n", argv[0]);
    return 1;
  }

  struct parameters param;
  if (read_parameters(&param, argv[1])) return 1;
  print_parameters(&param);

  struct data h;
  if (read_data(&h, param.input_h_filename)) return 1;

  // infer size of domain from input elevation data
  const double hx = h.nx * h.dx;
  const double hy = h.ny * h.dy;
  const int nx = max(1, floor(hx / param.dx));
  const int ny = max(1, floor(hy / param.dy));
  const int nt = floor(param.max_t / param.dt);

  printf(" - grid size: %g m x %g m (%d x %d = %d grid points)\n", hx, hy, nx, ny, nx * ny);
  printf(" - number of time steps: %d\n", nt);

  struct data eta, u, v;
  init_data(&eta, nx, ny, param.dx, param.dy, 0.);
  init_data(&u, nx + 1, ny, param.dx, param.dy, 0.);
  init_data(&v, nx, ny + 1, param.dx, param.dy, 0.);

  // interpolate bathymetry
  struct data h_interp;
  init_data(&h_interp, nx, ny, param.dx, param.dy, 0.);
#ifdef _OPENMP
  #pragma omp parallel for collapse(2)
#endif
  for (int j = 0; j < ny; j++)
    for (int i = 0; i < nx; i++)
      SET(&h_interp, i, j, interpolate_data(&h, i * param.dx, j * param.dy));

  const double start = GET_TIME();
  for (int n = 0; n < nt; n++) {

    if (n && (n % (nt / 10)) == 0) {
      const double time_sofar = GET_TIME() - start;
      const double eta = (nt - n) * time_sofar / n;
      printf("Computing step %d/%d (ETA: %g seconds)     \r", n, nt, eta);
      fflush(stdout);
    }

    // output solution
    if (param.sampling_rate && !(n % param.sampling_rate)) {
      write_data_vtk(&eta, "water elevation", param.output_eta_filename, n);
      //write_data_vtk(&u, "x velocity", param.output_u_filename, n);
      //write_data_vtk(&v, "y velocity", param.output_v_filename, n);
    }

    // impose boundary conditions
    const double t = n * param.dt;
    if (param.source_type == 1) {
      // sinusoidal velocity on top boundary
      const double A = 5;
      const double f = 1. / 20.;
      for (unsigned i = ny; i--;) {  // CHANGED (question 4)
        SET(&u, 0, i, 0.);
        SET(&u, nx, i, 0.);
      }
      for (unsigned i = nx; i--;) {
        SET(&v, i, 0, 0.);
        SET(&v, i, ny, A * sin(2 * M_PI * f * t));
      }
    } else if(param.source_type == 2) {
      // sinusoidal elevation in the middle of the domain
      const double A = 5;
      const double f = 1. / 20.;
      SET(&eta, nx / 2, ny / 2, A * sin(2 * M_PI * f * t));
    } else {
      // TODO: add other sources
      printf("Error: Unknown source type %d\n", param.source_type);
      exit(0);
    }
#ifdef _OPENMP
    #pragma omp parallel for collapse(2)
#endif
    for (int j = 0; j < ny; j++) {  // CHANGED (question 4)
      for (int i = 0; i < nx; i++) {
        // update eta
        // TODO: this does not evaluate h at the correct locations
        const double h_ij = GET(&h_interp, i, j);
        const double c0 = param.dt * h_ij;
        const double eta_ij = GET(&eta, i, j)
          - c0 / param.dx * (GET(&u, i + 1, j) - GET(&u, i, j))
          - c0 / param.dy * (GET(&v, i, j + 1) - GET(&v, i, j));
        SET(&eta, i, j, eta_ij);

        // update u and v
        const double c1 = param.dt * param.g;
        const double c2 = param.dt * param.gamma;
        const double eta_imj = i ? GET(&eta, i - 1, j) : eta_ij;
        const double eta_ijm = j ? GET(&eta, i, j - 1) : eta_ij;
        const double u_ij = (1. - c2) * GET(&u, i, j) - c1 / param.dx * (eta_ij - eta_imj);
        const double v_ij = (1. - c2) * GET(&v, i, j) - c1 / param.dy * (eta_ij - eta_ijm);
        SET(&u, i, j, u_ij);
        SET(&v, i, j, v_ij);
      }
    }
  }

  write_manifest_vtk(param.output_eta_filename, param.dt, nt, param.sampling_rate);
  //write_manifest_vtk(param.output_u_filename, param.dt, nt, param.sampling_rate);
  //write_manifest_vtk(param.output_v_filename, param.dt, nt, param.sampling_rate);

  const double time = GET_TIME() - start;
  printf("\nDone: %g seconds (%g MUpdates/s)\n", time, 1e-6 * (double)eta.nx * (double)eta.ny * (double)nt / time);

  free_data(&h_interp);
  free_data(&eta);
  free_data(&u);
  free_data(&v);

  return 0;
}
