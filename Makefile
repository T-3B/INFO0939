CC		= clang
OMPI_CC	= $(CC)
CFLAGS	= -O3 -flto -march=native -Wall -s -std=gnu2x -Wno-unknown-pragmas -Wno-unused-function
LDLIBS	= -lm
MPICC	= mpicc -DUSE_MPI
OMP		= -fopenmp
GPU		= -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda
SRC		= shallow.c
SRCGPU	= shallow_gpu.c
TARGET	= shallow

export OMPI_CC

all: mpi_omp mpi omp serial gpu

mpi_omp: $(TARGET)_mpi_omp
$(TARGET)_mpi_omp: $(SRC)
	$(MPICC) $(CFLAGS) $(OMP) -o $@ $^ $(LDLIBS)

mpi: $(TARGET)_mpi
$(TARGET)_mpi: $(SRC)
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDLIBS)

omp: $(TARGET)_omp
$(TARGET)_omp: $(SRC)
	$(CC) $(CFLAGS) $(OMP) -o $@ $^ $(LDLIBS)

serial: $(TARGET)_serial
$(TARGET)_serial: $(SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

gpu: $(TARGET)_gpu
$(TARGET)_gpu: $(SRCGPU)
	$(CC) $(CFLAGS) $(GPU) -o $@ $^ $(LDLIBS)

.PHONY: clean
clean:
	rm -f $(TARGET)_mpi_omp $(TARGET)_mpi $(TARGET)_omp $(TARGET)_serial $(TARGET)_gpu
