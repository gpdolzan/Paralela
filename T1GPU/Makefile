# CUDA compiler and flags
NVCC = nvcc
NVCC_FLAGS = -O3 -rdc=true -lcudadevrt # Adjust the architecture as per your GPU

# C compiler and flags (if you still have C code to compile)
CC = gcc
CFLAGS = -Wall -g -O3
LDLIBS = -lm -lpthread

# List of objects
objs = kSmallest.o

# Default target
all: kSmallest

# Rule to create the CUDA executable
kSmallest: $(objs)
	$(NVCC) $(NVCC_FLAGS) -o kSmallest $(objs)

# Rule to compile CUDA objects
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Rule to compile C objects (if you have any)
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Rule to clean temporary files
clean:
	-rm -f $(objs) *~

# Rule to clean everything (objects and executable)
purge: clean
	-rm -f kSmallest