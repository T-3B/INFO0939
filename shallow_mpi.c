#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

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

int read_parameters(struct parameters *param, const char *filename)
{
  FILE *fp = fopen(filename, "r");
  if(!fp) {
    printf("Error: Could not open parameter file '%s'\n", filename);
    return 1;
  }
  int ok = 1;
  if(ok) ok = (fscanf(fp, "%lf", &param->dx) == 1);
  if(ok) ok = (fscanf(fp, "%lf", &param->dy) == 1);
  if(ok) ok = (fscanf(fp, "%lf", &param->dt) == 1);
  if(ok) ok = (fscanf(fp, "%lf", &param->max_t) == 1);
  if(ok) ok = (fscanf(fp, "%lf", &param->g) == 1);
  if(ok) ok = (fscanf(fp, "%lf", &param->gamma) == 1);
  if(ok) ok = (fscanf(fp, "%d", &param->source_type) == 1);
  if(ok) ok = (fscanf(fp, "%d", &param->sampling_rate) == 1);
  if(ok) ok = (fscanf(fp, "%256s", param->input_h_filename) == 1);
  if(ok) ok = (fscanf(fp, "%256s", param->output_eta_filename) == 1);
  if(ok) ok = (fscanf(fp, "%256s", param->output_u_filename) == 1);
  if(ok) ok = (fscanf(fp, "%256s", param->output_v_filename) == 1);
  fclose(fp);
  if(!ok) {
    printf("Error: Could not read one or more parameters in '%s'\n", filename);
    return 1;
  }
  return 0;
}

int read_data(struct data *data, const char *filename)
{
  FILE *fp = fopen(filename, "rb");
  if(!fp) {
    printf("Error: Could not open input data file '%s'\n", filename);
    return 1;
  }
  int ok = 1;
  if(ok) ok = (fread(&data->nx, sizeof(int), 1, fp) == 1);
  if(ok) ok = (fread(&data->ny, sizeof(int), 1, fp) == 1);
  if(ok) ok = (fread(&data->dx, sizeof(double), 1, fp) == 1);
  if(ok) ok = (fread(&data->dy, sizeof(double), 1, fp) == 1);
  if(ok) {
    int N = data->nx * data->ny;
    if(N <= 0) {
      printf("Error: Invalid number of data points %d\n", N);
      ok = 0;
    }
    else {
      data->values = (double*)malloc(N * sizeof(double));
      if(!data->values) {
        printf("Error: Could not allocate data (%d doubles)\n", N);
        ok = 0;
      }
      else {
        ok = (fread(data->values, sizeof(double), N, fp) == N);
      }
    }
  }
  fclose(fp);
  if(!ok) {
    printf("Error reading input data file '%s'\n", filename);
    return 1;
  }
  return 0;
}

int write_data_vtk(const struct data *data, const char *name,
                   const char *filename, int step, int rank)
{
  char out[512];
  if(step < 0)
    sprintf(out, "%s_%d.vti", filename, rank);
  else
    sprintf(out, "%s_%d_%d.vti", filename, rank, step);

  FILE *fp = fopen(out, "wb");
  if(!fp) {
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
          "format=\"appended\" offset=\"0\">\n", name);
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

int init_data(struct data *data, int nx, int ny, double dx, double dy,
              double val)
{
  data->nx = nx;
  data->ny = ny;
  data->dx = dx;
  data->dy = dy;
  data->values = (double*)malloc(nx * ny * sizeof(double));
  if(!data->values){
    printf("Error: Could not allocate data\n");
    return 1;
  }
  for(int i = 0; i < nx * ny; i++) data->values[i] = val;
  return 0;
}

void free_data(struct data *data)
{
  free(data->values);
}

int main(int argc, char **argv)
{
  if(argc != 2) {
    printf("Usage: %s parameter_file\n", argv[0]);
    return 1;
  }

  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  struct parameters param;
  if(rank == 0) {
    if(read_parameters(&param, argv[1])) {
      MPI_Finalize();
      return 1;
    }
  }
  MPI_Bcast(&param, sizeof(struct parameters), MPI_BYTE, 0, MPI_COMM_WORLD);

  struct data h;
  if(rank == 0) {
    if(read_data(&h, param.input_h_filename)) {
      MPI_Finalize();
      return 1;
    }
  }

  MPI_Bcast(&h.nx, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&h.ny, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int local_ny = h.ny / size;
  int start_y = rank * local_ny;
  int end_y = start_y + local_ny;

  struct data local_eta, local_u, local_v;
  init_data(&local_eta, h.nx, local_ny, param.dx, param.dy, 0.);
  init_data(&local_u, h.nx + 1, local_ny, param.dx, param.dy, 0.);
  init_data(&local_v, h.nx, local_ny + 1, param.dx, param.dy, 0.);

  for(int t = 0; t < param.max_t / param.dt; t++) {
   
   // Update eta (Continuity equation)
    for (int i = 1; i < nx - 1; i++) {
        for (int j = 1; j < ny - 1; j++) {
            double hu_x = (GET(&h_interp, i + 1, j) * GET(&u, i + 1, j)) - (GET(&h_interp, i, j) * GET(&u, i, j));
            double hv_y = (GET(&h_interp, i, j + 1) * GET(&v, i, j + 1)) - (GET(&h_interp, i, j) * GET(&v, i, j));
            double eta_update = -(hu_x / param.dx) - (hv_y / param.dy);
            SET(&eta, i, j, GET(&eta, i, j) + param.dt * eta_update);
        }
    }

    // Impose boundary conditions (source term or other)
    double t = n * param.dt;
    if (param.source_type == 1) {
        // Sinusoidal boundary condition on top for v
        double A = 5;
        double f = 1. / 20.;
        for (int i = 0; i < nx; i++) {
            SET(&v, i, ny - 1, A * sin(2 * M_PI * f * t));
        }
    }

    // Update u (Momentum equation in x-direction)
    for (int i = 1; i < nx - 1; i++) {
        for (int j = 1; j < ny - 1; j++) {
            double eta_x = (GET(&eta, i, j) - GET(&eta, i - 1, j)) / param.dx;
            double u_update = -param.g * eta_x - param.gamma * GET(&u, i, j);
            SET(&u, i, j, GET(&u, i, j) + param.dt * u_update);
        }
    }

    // Update v (Momentum equation in y-direction)
    for (int i = 1; i < nx - 1; i++) {
        for (int j = 1; j < ny - 1; j++) {
            double eta_y = (GET(&eta, i, j) - GET(&eta, i, j - 1)) / param.dy;
            double v_update = -param.g * eta_y - param.gamma * GET(&v, i, j);
            SET(&v, i, j, GET(&v, i, j) + param.dt * v_update);
        }
    }

    if(param.sampling_rate && !(t % param.sampling_rate)) {
      write_data_vtk(&local_eta, "eta", param.output_eta_filename, t, rank);
      write_data_vtk(&local_u, "u", param.output_u_filename, t, rank);
      write_data_vtk(&local_v, "v", param.output_v_filename, t, rank);
    }
  }

  free_data(&local_eta);
  free_data(&local_u);
  free_data(&local_v);

  MPI_Finalize();

  return 0;
}

