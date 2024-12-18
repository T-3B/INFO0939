#include <libgen.h>
#include <limits.h>
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

# define ABORT() exit(EXIT_FAILURE)
# ifdef _OPENMP
#  define GET_TIME() (omp_get_wtime())  // wall time
# else
#  define GET_TIME() ((double)clock() / CLOCKS_PER_SEC)  // cpu time
# endif

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

// Declare the mapper for struct data
#pragma omp declare mapper(struct data mapper) \
    map(mapper)                                \
    map(mapper.values[0 : mapper.nx * mapper.ny])

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

static int init_data(struct data *const data, const int nx, const int ny, const double dx, const double dy, const double val) {
    data->nx = nx;
    data->ny = ny;
    data->dx = dx;
    data->dy = dy;

    // Allocate memory for 'values'
    data->values = malloc(nx * ny * sizeof(double));
    if (unlikely(!data->values)) {
        printf("Error: Could not allocate data\n");
        return 1;
    }
    for (unsigned i = 0; i < nx * ny; i++) {
        data->values[i] = val;
    }
    return 0;
}

void free_data(struct data *data){
  free(data->values);
}

// void interpolate_data(const struct data *h, const struct data *h_interp)
// {
//   #pragma omp target teams distribute
//   for (int j = 0; j < h_interp->ny; j++)
//   {
//     #pragma omp parallel for
//     for (int i = 0; i < h_interp->nx; i++)
//     {
//       int x = i * h_interp->dx;
//       int y = j * h_interp->dy;

//       int x_index = (int)(x / h->dx);
//       int y_index = (int)(y / h->dy);

//       if (x_index < 0)
//         x_index = 0;
//       if (x_index >= h->nx - 1)
//         x_index = h->nx - 2; // Ensure we have a valid neighbor
//       if (y_index < 0)
//         y_index = 0;
//       if (y_index >= h->ny - 1)
//         y_index = h->ny - 2;

//       double h_kl = GET(h, x_index, y_index);
//       double h_k1l = GET(h, x_index + 1, y_index);
//       double h_kl1 = GET(h, x_index, y_index + 1);
//       double h_k1l1 = GET(h, x_index + 1, y_index + 1);

//       // Get the four surrounding points
//       double x_k = x_index * h->dx;
//       double x_k1 = (x_index + 1) * h->dx;
//       double y_l = y_index * h->dy;
//       double y_l1 = (y_index + 1) * h->dy;

//       // Bilinear interpolation
//       double denom = (x_k1 - x_k) * (y_l1 - y_l);
//       double h_val = 1.0 / denom * (h_kl * (x_k1 - x) * (y_l1 - y) + h_k1l * (x - x_k) * (y_l1 - y) + h_kl1 * (x_k1 - x) * (y - y_l) + h_k1l1 * (x - x_k) * (y - y_l));
//       SET(h_interp, i, j, h_val);
//     }
//   }
// }

static void interpolate_data(const struct data *const interp, const struct data *const data, const int nx, const int ny, const double dx, const double dy) {
    // Trying to explain what i understood :
    // -> teams = group of threads working together on a part, so here we divide the computation into multiple teams.
    // -> collapse combines loops for better parallel performance.
    // -> map to/from: handles data transfer between the CPU and the GPU.

    // Parallelizing the interpolation calculation
    #pragma omp target teams distribute
    for (int jj = 0; jj < ny; jj++) {
        #pragma omp parallel for
        for (int ii = 0; ii < nx; ii++) {
            const double y = jj * dy;
            int j = (int)(y / data->dy);
            const double x = ii * dx;
            int i = (int)(x / data->dx);
            // Ensure indices are within bounds
            if (j < 0) j = 0;
            if (j >= data->ny - 1) j = data->ny - 2;
            if (i < 0) i = 0;
            if (i >= data->nx - 1) i = data->nx - 2;
            // Perform bilinear interpolation
            const double v00 = GET(data, i, j);
            const double v01 = GET(data, i, j + 1);
            const double v10 = GET(data, i + 1, j);
            const double v11 = GET(data, i + 1, j + 1);
            const double v0 = v00 + ((x - i * data->dx) / data->dx) * (v10 - v00);
            const double v1 = v01 + ((x - i * data->dx) / data->dx) * (v11 - v01);
            SET(interp, ii, jj, v0 + ((y - j * data->dy) / data->dy) * (v1 - v0));
        }
    }
}



int main(int argc, char **argv){
  if (argc != 2) {
    printf("Usage: %s parameter_file\n", argv[0]);
    return 1;
  }

  struct parameters param;
  if (read_parameters(&param, argv[1])) {
    return 1;
  }
  print_parameters(&param);

  struct data *h = malloc(sizeof(struct data));
  if (read_data(h, param.input_h_filename)) {
    return 1;
  }

  // Infer size of domain from input elevation data
  const double hx = h->nx * h->dx;
  const double hy = h->ny * h->dy;
  const int nx = max(1, floor(hx / param.dx));
  const int ny = max(1, floor(hy / param.dy));
  const int nt = floor(param.max_t / param.dt);

  printf(" - grid size: %g m x %g m (%d x %d = %d grid points)\n", hx, hy, nx, ny, nx * ny);
  printf(" - number of time steps: %d\n", nt);

  struct data *eta, *u, *v, *h_interp;
  eta = malloc(sizeof(struct data));
  u = malloc(sizeof(struct data));
  v = malloc(sizeof(struct data));
  h_interp = malloc(sizeof(struct data));
  init_data(eta, nx, ny, param.dx, param.dy, 0.0);
  init_data(u, nx + 1, ny, param.dx, param.dy, 0.0);
  init_data(v, nx, ny + 1, param.dx, param.dy, 0.0);
  init_data(h_interp, nx, ny, param.dx, param.dy, 0.0);
  const double start = GET_TIME();
  #pragma omp target data map(to: u[0:1], v[0:1], h_interp[0:1], h[0:1], eta[0:1]){
    // Interpolate bathymetry using interpolate_data
    interpolate_data(h_interp, h, nx, ny, param.dx, param.dy);
    for (int n = 0; n < nt; n++) {

      // if (n && (n % (nt / 10)) == 0) {
      //   const double time_sofar = GET_TIME() - start;
      //   const double eta = (nt - n) * time_sofar / n;
      //   printf("Computing step %d/%d (ETA: %g seconds)     \r", n, nt, eta);
      //   fflush(stdout);
      // }

      // Output solution
      if (param.sampling_rate && !(n % param.sampling_rate)) {
        #pragma omp target update from(eta[0:1])
        write_data_vtk(eta, "water elevation", param.output_eta_filename, n);
      }

      const double t = n * param.dt;
      if (param.source_type == 1) {
        const double A = 5;
        const double f = 1. / 20.;

        #pragma omp target teams distribute
        for (int i = 0; i < ny; i++) {
          SET(u, 0, i, 0.0);
          SET(u, nx, i, 0.0);
        }
        #pragma omp target teams distribute
        for (int i = 0; i < nx; i++) {
          SET(v, i, 0, 0.0);
          SET(v, i, ny, A * sin(2 * M_PI * f * t));
        }
      } else if (param.source_type == 2) {
        const double A = 5;
        const double f = 1. / 20.;
        SET(eta, nx / 2, ny / 2, A * sin(2 * M_PI * f * t));
      } else {
        printf("Error: Unknown source type %d\n", param.source_type);
        
      }

      #pragma omp target teams distribute
      for (int j = 0; j < ny; j++) { 
        #pragma omp parallel for
        for (int i = 0; i < nx; i++) {
          // Update eta
          const double h_ij = GET(h_interp, i, j);
          double u_ij = GET(u, i, j);
          double v_ij = GET(v, i, j);
          const double eta_ij = GET(eta, i, j) - param.dt * (
            (GET(h_interp, i + 1, j) * GET(u, i + 1, j) - h_ij * u_ij) / param.dx
            + (GET(h_interp, i, j + 1) * GET(v, i, j + 1) - h_ij * v_ij) / param.dy);
          SET(eta, i, j, eta_ij);

          // Update u and v
          const double c1 = param.dt * param.g;
          const double c2 = param.dt * param.gamma;
          const double eta_imj = i ? GET(eta, i - 1, j) : eta_ij;
          const double eta_ijm = j ? GET(eta, i, j - 1) : eta_ij;
          u_ij = (1. - c2) * u_ij - c1 / param.dx * (eta_ij - eta_imj);
          v_ij = (1. - c2) * v_ij - c1 / param.dy * (eta_ij - eta_ijm);
          SET(u, i, j, u_ij);
          SET(v, i, j, v_ij);
        }
      }
    }
  

  write_manifest_vtk(param.output_eta_filename, param.dt, nt, param.sampling_rate);

  const double time = GET_TIME() - start;
  printf("\nDone: %g seconds (%g MUpdates/s)\n", time,
         1e-6 * (double)eta->nx * (double)eta->ny * (double)nt / time);

  free_data(h_interp);
  free_data(eta);
  free_data(u);
  free_data(v);
  free_data(h);
  free(h);
  free(h_interp);
  free(eta);
  free(u);
  free(v);

  return 0;
}