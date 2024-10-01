#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>

#if defined(_OPENMP)
#include <omp.h>
#define GET_TIME() (omp_get_wtime()) // wall time
#else
#define GET_TIME() ((double)clock() / CLOCKS_PER_SEC) // cpu time
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

int read_parameters(struct parameters *param, const char *filename) {
  FILE *fp = fopen(filename, "r");
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

void print_parameters(const struct parameters *param) {
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
  printf(" - output velocity (u, v) files: '%s', '%s'\n",
         param->output_u_filename, param->output_v_filename);
}

int read_data(struct data *data, const char *filename) {
  FILE *fp = fopen(filename, "rb");
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
    int N = data->nx * data->ny;
    if (N <= 0) {
      printf("Error: Invalid number of data points %d\n", N);
      ok = 0;
    } else {
      // Use aligned_alloc for optimized memory access
      data->values = (double *)aligned_alloc(64, N * sizeof(double));
      if (!data->values) {
        printf("Error: Could not allocate data (%d doubles)\n", N);
        ok = 0;
      } else {
        ok = (fread(data->values, sizeof(double), N, fp) == N);
      }
    }
  }
  fclose(fp);
  if (!ok) {
    printf("Error reading input data file '%s'\n", filename);
    return 1;
  }
  return 0;
}

int write_data(const struct data *data, const char *filename, int step) {
  char out[512];
  if (step < 0)
    sprintf(out, "%s.dat", filename);
  else
    sprintf(out, "%s_%d.dat", filename, step);
  FILE *fp = fopen(out, "wb");
  if (!fp) {
    printf("Error: Could not open output data file '%s'\n", out);
    return 1;
  }
  int ok = 1;
  if (ok) ok = (fwrite(&data->nx, sizeof(int), 1, fp) == 1);
  if (ok) ok = (fwrite(&data->ny, sizeof(int), 1, fp) == 1);
  if (ok) ok = (fwrite(&data->dx, sizeof(double), 1, fp) == 1);
  if (ok) ok = (fwrite(&data->dy, sizeof(double), 1, fp) == 1);
  int N = data->nx * data->ny;
  if (ok) ok = (fwrite(data->values, sizeof(double), N, fp) == N);
  fclose(fp);
  if (!ok) {
    printf("Error writing data file '%s'\n", out);
    return 1;
  }
  return 0;
}

int init_data(struct data *data, int nx, int ny, double dx, double dy, double val) {
  data->nx = nx;
  data->ny = ny;
  data->dx = dx;
  data->dy = dy;
  data->values = (double *)aligned_alloc(64, nx * ny * sizeof(double));  // aligned memory
  if (!data->values) {
    printf("Error: Could not allocate data\n");
    return 1;
  }
  
  // Parallelize initialization
  #pragma omp parallel for
  for (int i = 0; i < nx * ny; i++) {
    data->values[i] = val;
  }
  return 0;
}

void free_data(struct data *data) {
  free(data->values);
}

double interpolate_data(const struct data *data, double x, double y) {
  int i = (int)(x / data->dx);  // Nearest integer from origin
  int j = (int)(y / data->dy);

  if (i < 0) i = 0;
  else if (i >= data->nx - 1) i = data->nx - 2;
  if (j < 0) j = 0;
  else if (j >= data->ny - 1) j = data->ny - 2;

  double v00 = GET(data, i, j);
  double v10 = GET(data, i + 1, j);
  double v01 = GET(data, i, j + 1);
  double v11 = GET(data, i + 1, j + 1);

  double v0 = v00 + ((x - i * data->dx) / data->dx) * (v10 - v00);
  double v1 = v01 + ((x - i * data->dx) / data->dx) * (v11 - v01);

  return v0 + ((y - j * data->dy) / data->dy) * (v1 - v0);
}

void simulate(const struct parameters *params, const struct data *h, struct data *eta, struct data *u, struct data *v) {
  double g = params->g;
  double gamma = params->gamma;
  double t = 0;
  int step = 0;

  double dt = params->dt;
  double dx = params->dx;
  double dy = params->dy;
  
  double max_t = params->max_t;

  while (t < max_t) {
    // Time-stepping loop parallelized
    #pragma omp parallel for
    for (int j = 1; j < eta->ny - 1; j++) {
      for (int i = 1; i < eta->nx - 1; i++) {
        double hval = GET(h, i, j);
        double etaval = GET(eta, i, j);
        double uval = GET(u, i, j);
        double vval = GET(v, i, j);

        // Euler-forward update for eta
        double du_dx = (GET(u, i + 1, j) - GET(u, i - 1, j)) / (2.0 * dx);
        double dv_dy = (GET(v, i, j + 1) - GET(v, i, j - 1)) / (2.0 * dy);
        SET(eta, i, j, etaval - dt * (du_dx + dv_dy));

        // Euler-forward update for u, v
        double deta_dx = (GET(eta, i + 1, j) - GET(eta, i - 1, j)) / (2.0 * dx);
        double deta_dy = (GET(eta, i, j + 1) - GET(eta, i, j - 1)) / (2.0 * dy);

        SET(u, i, j, uval - dt * g * deta_dx - dt * gamma * uval);
        SET(v, i, j, vval - dt * g * deta_dy - dt * gamma * vval);
      }
    }
    
    t += dt;
    step++;
  }
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("Usage: %s parameter_file\n", argv[0]);
    return 1;
  }
  const char *param_filename = argv[1];

  struct parameters param;
  if (read_parameters(&param, param_filename)) return 1;
  print_parameters(&param);

  struct data h, eta, u, v;
  if (read_data(&h, param.input_h_filename)) return 1;
  if (init_data(&eta, h.nx, h.ny, h.dx, h.dy, 0.0)) return 1;
  if (init_data(&u, h.nx, h.ny, h.dx, h.dy, 0.0)) return 1;
  if (init_data(&v, h.nx, h.ny, h.dx, h.dy, 0.0)) return 1;

  double start = GET_TIME();
  simulate(&param, &h, &eta, &u, &v);
  double end = GET_TIME();
  printf("Elapsed time: %g s\n", end - start);

  free_data(&h);
  free_data(&eta);
  free_data(&u);
  free_data(&v);

  return 0;
}

