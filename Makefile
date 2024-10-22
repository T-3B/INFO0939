CC		= clang
CFLAGS	= -O3 -flto -march=native -Wall -s -std=gnu23 -Wno-unknown-pragmas
LDLIBS	= -lm
MPICC	= mpicc
OMP		= -fopenmp
SRC		= shallow.c
TARGET	= shallow

all: omp_mpi mpi omp serial

omp_mpi: $(TARGET)
$(TARGET): $(SRC)
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

.PHONY: clean
clean:
	rm -f $(TARGET) $(TARGET)_mpi $(TARGET)_omp $(TARGET)_serial
