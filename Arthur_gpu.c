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

struct parameters
{
  double dx, dy, dt, max_t;
  double g, gamma;
  int source_type;
  int sampling_rate;
  char input_h_filename[256];
  char output_eta_filename[256];
  char output_u_filename[256];
  char output_v_filename[256];
};

struct data
{
  int nx, ny;
  double dx, dy;
  double *values;
};

// Declare the mapper for struct data
#pragma omp declare mapper(struct data mapper_data) \
    map(mapper_data)                                \
    map(mapper_data.values[0 : mapper_data.nx * mapper_data.ny])

#define GET(data, i, j) ((data)->values[(data)->nx * (j) + (i)])
#define SET(data, i, j, val) ((data)->values[(data)->nx * (j) + (i)] = (val))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

int read_parameters(struct parameters *param, const char *filename)
{
  FILE *fp = fopen(filename, "r");
  if (!fp)
  {
    printf("Error: Could not open parameter file '%s'\n", filename);
    return 1;
  }
  int ok = 1;
  if (ok)
    ok = (fscanf(fp, "%lf", &param->dx) == 1);
  if (ok)
    ok = (fscanf(fp, "%lf", &param->dy) == 1);
  if (ok)
    ok = (fscanf(fp, "%lf", &param->dt) == 1);
  if (ok)
    ok = (fscanf(fp, "%lf", &param->max_t) == 1);
  if (ok)
    ok = (fscanf(fp, "%lf", &param->g) == 1);
  if (ok)
    ok = (fscanf(fp, "%lf", &param->gamma) == 1);
  if (ok)
    ok = (fscanf(fp, "%d", &param->source_type) == 1);
  if (ok)
    ok = (fscanf(fp, "%d", &param->sampling_rate) == 1);
  if (ok)
    ok = (fscanf(fp, "%256s", param->input_h_filename) == 1);
  if (ok)
    ok = (fscanf(fp, "%256s", param->output_eta_filename) == 1);
  if (ok)
    ok = (fscanf(fp, "%256s", param->output_u_filename) == 1);
  if (ok)
    ok = (fscanf(fp, "%256s", param->output_v_filename) == 1);
  fclose(fp);
  if (!ok)
  {
    printf("Error: Could not read one or more parameters in '%s'\n", filename);
    return 1;
  }
  return 0;
}

void print_parameters(const struct parameters *param)
{
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

int read_data(struct data *data, const char *filename)
{
  FILE *fp = fopen(filename, "rb");
  if (!fp)
  {
    printf("Error: Could not open input data file '%s'\n", filename);
    return 1;
  }
  int ok = 1;
  if (ok)
    ok = (fread(&data->nx, sizeof(int), 1, fp) == 1);
  if (ok)
    ok = (fread(&data->ny, sizeof(int), 1, fp) == 1);
  if (ok)
    ok = (fread(&data->dx, sizeof(double), 1, fp) == 1);
  if (ok)
    ok = (fread(&data->dy, sizeof(double), 1, fp) == 1);
  if (ok)
  {
    int N = data->nx * data->ny;
    if (N <= 0)
    {
      printf("Error: Invalid number of data points %d\n", N);
      ok = 0;
    }
    else
    {
      data->values = (double *)malloc(N * sizeof(double));
      if (!data->values)
      {
        printf("Error: Could not allocate data (%d doubles)\n", N);
        ok = 0;
      }
      else
      {
        ok = (fread(data->values, sizeof(double), N, fp) == N);
      }
    }
  }
  fclose(fp);
  if (!ok)
  {
    printf("Error reading input data file '%s'\n", filename);
    return 1;
  }
  return 0;
}

int write_data(const struct data *data, const char *filename, int step)
{
  char out[512];
  if (step < 0)
    sprintf(out, "%s.dat", filename);
  else
    sprintf(out, "%s_%d.dat", filename, step);
  FILE *fp = fopen(out, "wb");
  if (!fp)
  {
    printf("Error: Could not open output data file '%s'\n", out);
    return 1;
  }
  int ok = 1;
  if (ok)
    ok = (fwrite(&data->nx, sizeof(int), 1, fp) == 1);
  if (ok)
    ok = (fwrite(&data->ny, sizeof(int), 1, fp) == 1);
  if (ok)
    ok = (fwrite(&data->dx, sizeof(double), 1, fp) == 1);
  if (ok)
    ok = (fwrite(&data->dy, sizeof(double), 1, fp) == 1);
  int N = data->nx * data->ny;
  if (ok)
    ok = (fwrite(data->values, sizeof(double), N, fp) == N);
  fclose(fp);
  if (!ok)
  {
    printf("Error writing data file '%s'\n", out);
    return 1;
  }
  return 0;
}

int write_data_vtk(const struct data *data, const char *name,
                   const char *filename, int step)
{
  char out[512];
  if (step < 0)
    sprintf(out, "%s.vti", filename);
  else
    sprintf(out, "%s_%d.vti", filename, step);

  FILE *fp = fopen(out, "wb");
  if (!fp)
  {
    printf("Error: Could not open output VTK file '%s'\n", out);
    return 1;
  }

  unsigned long num_points = data->nx * data->ny;
  unsigned long num_bytes = num_points * sizeof(double);

  fprintf(fp, "<?xml version=\"1.0\"?>\n");
  fprintf(fp, "<VTKFile type=\"ImageData\" version=\"1.0\" "
              "byte_order=\"LittleEndian\" header_type=\"UInt64\">\n");
  fprintf(fp, "  <ImageData WholeExtent=\"0 %d 0 %d 0 0\" "
              "Spacing=\"%lf %lf 0.0\">\n",
          data->nx - 1, data->ny - 1, data->dx, data->dy);
  fprintf(fp, "    <Piece Extent=\"0 %d 0 %d 0 0\">\n",
          data->nx - 1, data->ny - 1);

  fprintf(fp, "      <PointData Scalars=\"scalar_data\">\n");
  fprintf(fp, "        <DataArray type=\"Float64\" Name=\"%s\" "
              "format=\"appended\" offset=\"0\">\n",
          name);
  fprintf(fp, "        </DataArray>\n");
  fprintf(fp, "      </PointData>\n");

  fprintf(fp, "    </Piece>\n");
  fprintf(fp, "  </ImageData>\n");

  fprintf(fp, "  <AppendedData encoding=\"raw\">\n_");

  fwrite(&num_bytes, sizeof(unsigned long), 1, fp);
  fwrite(data->values, sizeof(double), num_points, fp);

  fprintf(fp, "  </AppendedData>\n");
  fprintf(fp, "</VTKFile>\n");

  fclose(fp);
  return 0;
}

int write_manifest_vtk(const char *name, const char *filename,
                       double dt, int nt, int sampling_rate)
{
  char out[512];
  sprintf(out, "%s.pvd", filename);

  FILE *fp = fopen(out, "wb");
  if (!fp)
  {
    printf("Error: Could not open output VTK manifest file '%s'\n", out);
    return 1;
  }

  fprintf(fp, "<VTKFile type=\"Collection\" version=\"0.1\" "
              "byte_order=\"LittleEndian\">\n");
  fprintf(fp, "  <Collection>\n");
  for (int n = 0; n < nt; n++)
  {
    if (sampling_rate && !(n % sampling_rate))
    {
      double t = n * dt;
      fprintf(fp, "    <DataSet timestep=\"%g\" file='%s_%d.vti'/>\n", t,
              filename, n);
    }
  }
  fprintf(fp, "  </Collection>\n");
  fprintf(fp, "</VTKFile>\n");
  fclose(fp);
  return 0;
}

int init_data(struct data *data, int nx, int ny, double dx, double dy,
              double val)
{
  data->nx = nx;
  data->ny = ny;
  data->dx = dx;
  data->dy = dy;
  data->values = (double *)malloc(nx * ny * sizeof(double));
  if (!data->values)
  {
    printf("Error: Could not allocate data\n");
    return 1;
  }
  for (int i = 0; i < nx * ny; i++)
    data->values[i] = val;
  return 0;
}

void free_data(struct data *data)
{
  free(data->values);
}

void interpolate_data(const struct data *h, const struct data *h_interp)
{
  #pragma omp target teams distribute
  for (int j = 0; j < h_interp->ny; j++)
  {
    #pragma omp parallel for
    for (int i = 0; i < h_interp->nx; i++)
    {
      int x = i * h_interp->dx;
      int y = j * h_interp->dy;

      int x_index = (int)(x / h->dx);
      int y_index = (int)(y / h->dy);

      if (x_index < 0)
        x_index = 0;
      if (x_index >= h->nx - 1)
        x_index = h->nx - 2; // Ensure we have a valid neighbor
      if (y_index < 0)
        y_index = 0;
      if (y_index >= h->ny - 1)
        y_index = h->ny - 2;

      double h_kl = GET(h, x_index, y_index);
      double h_k1l = GET(h, x_index + 1, y_index);
      double h_kl1 = GET(h, x_index, y_index + 1);
      double h_k1l1 = GET(h, x_index + 1, y_index + 1);

      // Get the four surrounding points
      double x_k = x_index * h->dx;
      double x_k1 = (x_index + 1) * h->dx;
      double y_l = y_index * h->dy;
      double y_l1 = (y_index + 1) * h->dy;

      // Bilinear interpolation
      double denom = (x_k1 - x_k) * (y_l1 - y_l);
      double h_val = 1.0 / denom * (h_kl * (x_k1 - x) * (y_l1 - y) + h_k1l * (x - x_k) * (y_l1 - y) + h_kl1 * (x_k1 - x) * (y - y_l) + h_k1l1 * (x - x_k) * (y - y_l));
      SET(h_interp, i, j, h_val);
    }
  }
}

int main(int argc, char **argv)
{
  if (argc != 2)
  {
    printf("Usage: %s parameter_file\n", argv[0]);
    return 1;
  }

  struct parameters param;
  if (read_parameters(&param, argv[1]))
    return 1;
  print_parameters(&param);

  struct data *h = malloc(sizeof(struct data));
  if (read_data(h, param.input_h_filename))
    return 1;

  // infer size of domain from input elevation data
  double hx = h->nx * h->dx;
  double hy = h->ny * h->dy;
  int nx = floor(hx / param.dx);
  int ny = floor(hy / param.dy);
  if (nx <= 0)
    nx = 1;
  if (ny <= 0)
    ny = 1;
  int nt = floor(param.max_t / param.dt);

  printf(" - grid size: %g m x %g m (%d x %d = %d grid points)\n",
         hx, hy, nx, ny, nx * ny);
  printf(" - number of time steps: %d\n", nt);

  struct data *eta, *u, *v;
  eta = malloc(sizeof(struct data));
  u = malloc(sizeof(struct data));
  v = malloc(sizeof(struct data));
  init_data(eta, nx, ny, param.dx, param.dy, 0.);
  init_data(u, nx + 1, ny, param.dx, param.dy, 0.);
  init_data(v, nx, ny + 1, param.dx, param.dy, 0.);

  // interpolate bathymetry
  struct data *h_interp = malloc(sizeof(struct data));
  init_data(h_interp, nx, ny, param.dx, param.dy, 0.);

  double start = GET_TIME();

#pragma omp target data map(to: u[0:1], \
                                v[0:1], \
                                h_interp[0:1], \
                                h[0:1], \
                                eta[0:1])
  {

    interpolate_data(h, h_interp);

    for (int n = 0; n < nt; n++)
    {

      /*
      if (n && (n % (nt / 10)) == 0)
      {
        double time_sofar = GET_TIME() - start;
        double eta = (nt - n) * time_sofar / n;
        printf("Computing step %d/%d (ETA: %g seconds)     \r", n, nt, eta);
        fflush(stdout);
      }
      */

      // output solution
      if (param.sampling_rate && !(n % param.sampling_rate))
      {
        #pragma omp target update from(eta[0:1])
        write_data_vtk(eta, "water elevation", param.output_eta_filename, n);
        // write_data_vtk(&u, "x velocity", param.output_u_filename, n);
        // write_data_vtk(&v, "y velocity", param.output_v_filename, n);
      }

      // impose boundary conditions
      double t = n * param.dt;
      if (param.source_type == 1)
      {
        // sinusoidal velocity on top boundary
        double A = 5;
        double f = 1. / 20.;
        #pragma omp target teams distribute
        for (int j = 0; j < ny; j++)
        {
          #pragma omp parallel for
          for (int i = 0; i < nx; i++)
          {
            SET(u, 0, j, 0.);
            SET(u, nx, j, 0.);
            SET(v, i, 0, 0.);
            SET(v, i, ny, A * sin(2 * M_PI * f * t));
          }
        }
      }
      else if (param.source_type == 2)
      {
        // sinusoidal elevation in the middle of the domain
        double A = 5;
        double f = 1. / 20.;
        SET(eta, nx / 2, ny / 2, A * sin(2 * M_PI * f * t));
      }
      else
      {
        // TODO: add other sources
        printf("Error: Unknown source type %d\n", param.source_type);
        exit(0);
      }

// update eta
      #pragma omp target teams distribute
      for (int j = 0; j < ny; j++)
      {
        #pragma omp parallel for
        for (int i = 0; i < nx; i++)
        {
          double h_ij = GET(h_interp, i, j);
          double h_i1j = GET(h_interp, MIN(i + 1, nx - 1), j);
          double h_ij1 = GET(h_interp, i, MIN(j + 1, ny - 1));
          double c0 = param.dt * h_ij;
          double c1 = param.dt * h_i1j;
          double c2 = param.dt * h_ij1;

          double eta_ij = GET(eta, i, j) - (c1 * GET(u, i + 1, j) - c0 * GET(u, i, j)) / param.dx - (c2 * GET(v, i, j + 1) - c0 * GET(v, i, j)) / param.dy;
          SET(eta, i, j, eta_ij);
        }
      }

// update u and v
      #pragma omp target teams distribute
      for (int j = 0; j < ny; j++)
      {
        #pragma omp parallel for
        for (int i = 0; i < nx; i++)
        {
          double c1 = param.dt * param.g;
          double c2 = param.dt * param.gamma;
          double eta_ij = GET(eta, i, j);
          double eta_imj = GET(eta, (i == 0) ? 0 : i - 1, j);
          double eta_ijm = GET(eta, i, (j == 0) ? 0 : j - 1);
          double u_ij = (1. - c2) * GET(u, i, j) - c1 / param.dx * (eta_ij - eta_imj);
          double v_ij = (1. - c2) * GET(v, i, j) - c1 / param.dy * (eta_ij - eta_ijm);
          SET(u, i, j, u_ij);
          SET(v, i, j, v_ij);
        }
      }
    }
  }

  write_manifest_vtk("water elevation", param.output_eta_filename,
                     param.dt, nt, param.sampling_rate);
  // write_manifest_vtk("x velocity", param.output_u_filename,
  //                    param.dt, nt, param.sampling_rate);
  // write_manifest_vtk("y velocity", param.output_v_filename,
  //                    param.dt, nt, param.sampling_rate);

  double time = GET_TIME() - start;
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