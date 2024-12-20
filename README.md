# INFO 0939 - High Performance Scientific Computing
This project was done by HANSEN Julien and SMAGGHE Clément.

To test this project, a simple `make -j` will build all targets, that are:
- serial (mono-CPU, with fixed code for better cache utilization)
- omp
- mpi
- mpi_omp (hybrid)
- gpu (mono)

All targets use the file `shallow.c` except for the target `gpu` which uses `shallow_gpu.c`.

Note that we chose to implement transparent boundary conditions (available with `param.source_type = 3`) and Coriolis forces (used on every `param.source_type`).
We also decided to look at large simulations with realistic bathymetry files, that we will talk about in the future presentation.
