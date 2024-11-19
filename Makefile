CC		= clang
OMPI_CC	= $(CC)
CFLAGS	= -O3 -flto -march=native -Wall -s -std=gnu23 -Wno-unknown-pragmas
LDLIBS	= -lm
MPICC	= mpicc -DUSE_MPI
OMP		= -fopenmp
SRC		= shallow.c
MPISRC	= shallow_mpi.c
TARGET	= shallow

export OMPI_CC

all: omp_mpi mpi omp serial

omp_mpi: $(TARGET)
$(TARGET): $(MPISRC)
	$(MPICC) $(CFLAGS) $(OMP) -o $@ $^ $(LDLIBS)

mpi: $(TARGET)_mpi
$(TARGET)_mpi: $(MPISRC)
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDLIBS)

omp: $(TARGET)_omp
$(TARGET)_omp: $(SRC)
	$(CC) $(CFLAGS) $(OMP) -o $@ $^ $(LDLIBS)

serial: $(TARGET)_serial
$(TARGET)_serial: $(SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

.PHONY: clean
clean:
	rm -f $(TARGET) $(TARGET)_mpi $(TARGET)_omp $(TARGET)_serial
