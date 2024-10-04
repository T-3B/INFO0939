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

int write_data(const struct data *data, const char *filename, int step)
{
  char out[512];
  if(step < 0)
    sprintf(out, "%s.dat", filename);
  else
    sprintf(out, "%s_%d.dat", filename, step);
  FILE *fp = fopen(out, "wb");
  if(!fp) {
    printf("Error: Could not open output data file '%s'\n", out);
    return 1;
  }
  int ok = 1;
  if(ok) ok = (fwrite(&data->nx, sizeof(int), 1, fp) == 1);
  if(ok) ok = (fwrite(&data->ny, sizeof(int), 1, fp) == 1);
  if(ok) ok = (fwrite(&data->dx, sizeof(double), 1, fp) == 1);
  if(ok) ok = (fwrite(&data->dy, sizeof(double), 1, fp) == 1);
  int N = data->nx * data->ny;
  if(ok) ok = (fwrite(data->values, sizeof(double), N, fp) == N);
  fclose(fp);
  if(!ok) {
    printf("Error writing data file '%s'\n", out);
    return 1;
  }
  return 0;
}

int write_data_vtk(const struct data *data, const char *name, const char *filename, int step)
{
  char out[512];
  if(step < 0)
    sprintf(out, "%s.vti", filename);
  else
    sprintf(out, "%s_%d.vti", filename, step);

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

int write_manifest_vtk(const char *name, const char *filename, double dt, int nt, int sampling_rate)
{
  char out[512];
  sprintf(out, "%s.pvd", filename);

  FILE *fp = fopen(out, "wb");
  if(!fp) {
    printf("Error: Could not open output VTK manifest file '%s'\n", out);
    return 1;
  }

  fprintf(fp, "<VTKFile type=\"Collection\" version=\"0.1\" "
          "byte_order=\"LittleEndian\">\n");
  fprintf(fp, "  <Collection>\n");
  for(int n = 0; n < nt; n++) {
    if(sampling_rate && !(n % sampling_rate)) {
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

double interpolate_data(const struct data *data, double x, double y)
{  // TODO could store GET values between calls (multiple points could be in the same small square v00 v01 v10 v11)
  int i = (int) (x / data->dx);
  int j = (int) (y / data->dy);
  
  if(i < 0) i = 0;
  else if(i > data->nx - 1) i = data->nx - 1;
  if(j < 0) j = 0;
  else if(j > data->ny - 1) j = data->ny - 1;

  double v00 = GET(data, i, j);         // Top-left
  double v10 = GET(data, i + 1, j);     // Top-right
  double v01 = GET(data, i, j + 1);     // Bottom-left
  double v11 = GET(data, i + 1, j + 1); // Bottom-right  
  // Interpolate along the x direction
  double v0 = v00 + ((x - i * data->dx) / data->dx) * (v10 - v00);
  double v1 = v01 + ((x - i * data->dx) / data->dx) * (v11 - v01);
  // Interpolate along the y direction and return the result
  return (double) (v0 + ((y - j * data->dy) / data->dy) * (v1 - v0));
}


int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2) {
        if (rank == 0) printf("Usage: %s parameter_file\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    struct parameters param;
    if (rank == 0) {
        if (read_parameters(&param, argv[1])) {
            MPI_Finalize();
            return 1;
        }
    }
    MPI_Bcast(&param, sizeof(struct parameters), MPI_BYTE, 0, MPI_COMM_WORLD);

    struct data h;
    if (rank == 0) {
        if (read_data(&h, param.input_h_filename)) {
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Bcast(&h.nx, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&h.ny, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int local_nx = h.nx / size;
    int local_ny = h.ny;

    struct data local_eta, local_u, local_v, local_h_interp;
    init_data(&local_eta, local_nx, local_ny, param.dx, param.dy, 0.0);
    init_data(&local_u, local_nx + 1, local_ny, param.dx, param.dy, 0.0);
    init_data(&local_v, local_nx, local_ny + 1, param.dx, param.dy, 0.0);
    init_data(&local_h_interp, local_nx, local_ny, param.dx, param.dy, 0.0);

    // Interpolate bathymetry
    for (int j = 0; j < local_ny; j++) {
        for (int i = 0; i < local_nx; i++) {
            double x = (rank * local_nx + i) * param.dx;
            double y = j * param.dy;
            double val = interpolate_data(&h, x, y);
            SET(&local_h_interp, i, j, val);
        }
    }

    double start = MPI_Wtime();
    int nt = floor(param.max_t / param.dt);

    for (int n = 0; n < nt; n++) {
        if (n && (n % (nt / 10)) == 0 && rank == 0) {
            double time_sofar = MPI_Wtime() - start;
            double eta = (nt - n) * time_sofar / n;
            printf("Computing step %d/%d (ETA: %g seconds)     \r", n, nt, eta);
            fflush(stdout);
        }

        // Impose boundary conditions
        double t = n * param.dt;
        if (param.source_type == 1) {
            double A = 5;
            double f = 1. / 20.;
            for (int i = 0; i < local_nx; i++) {
                SET(&local_v, i, local_ny - 1, A * sin(2 * M_PI * f * t));
            }
        }

        // Update eta (continuity equation)
        for (int i = 1; i < local_nx - 1; i++) {
            for (int j = 1; j < local_ny - 1; j++) {
                double hu_x = (GET(&local_h_interp, i + 1, j) * GET(&local_u, i + 1, j)) - (GET(&local_h_interp, i, j) * GET(&local_u, i, j));
                double hv_y = (GET(&local_h_interp, i, j + 1) * GET(&local_v, i, j + 1)) - (GET(&local_h_interp, i, j) * GET(&local_v, i, j));
                double eta_update = -(hu_x / param.dx) - (hv_y / param.dy);
                SET(&local_eta, i, j, GET(&local_eta, i, j) + param.dt * eta_update);
            }
        }

        // Update u (momentum equation in x-direction)
        for (int i = 1; i < local_nx - 1; i++) {
            for (int j = 1; j < local_ny - 1; j++) {
                double eta_x = (GET(&local_eta, i, j) - GET(&local_eta, i - 1, j)) / param.dx;
                double u_update = -param.g * eta_x - param.gamma * GET(&local_u, i, j);
                SET(&local_u, i, j, GET(&local_u, i, j) + param.dt * u_update);
            }
        }

        // Update v (momentum equation in y-direction)
        for (int i = 1; i < local_nx - 1; i++) {
            for (int j = 1; j < local_ny - 1; j++) {
                double eta_y = (GET(&local_eta, i, j) - GET(&local_eta, i, j - 1)) / param.dy;
                double v_update = -param.g * eta_y - param.gamma * GET(&local_v, i, j);
                SET(&local_v, i, j, GET(&local_v, i, j) + param.dt * v_update);
            }
        }

        // Output local data periodically
        if (param.sampling_rate && !(n % param.sampling_rate)) {
	  write_data_vtk(&local_eta, "water elevation", param.output_eta_filename, n);
          //write_data_vtk(&local_u, "x velocity", param.output_u_filename, n);
          //write_data_vtk(&local_v, "y velocity", param.output_v_filename, n);
        }
    }

    write_manifest_vtk("water elevation", param.output_eta_filename, param.dt, nt, param.sampling_rate);

    free_data(&local_h_interp);
    free_data(&local_eta);
    free_data(&local_u);
    free_data(&local_v);

    MPI_Finalize();

    return 0;
}
