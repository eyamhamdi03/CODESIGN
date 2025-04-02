
from helper import *
from definitions import *

import numpy

import pyopencl as cl

from time import time
from time import sleep


# A[N][N], B[N][N], C[N][N]
N =8192

# Number of elements in the matrix
size = N * N
#true value
cval = float(N) * AVAL * BVAL

# A matrix
h_A = numpy.empty(size).astype(numpy.float32)
h_A.fill(AVAL)

# B matrix
h_B = numpy.empty(size).astype(numpy.float32)
h_B.fill(BVAL)

# C matrix
h_C = numpy.empty(size).astype(numpy.float32)


#--------------------------------------------------------------------
# Explicitly select NVIDIA GPU
#--------------------------------------------------------------------
platforms = cl.get_platforms()
nvidia_platform = next(p for p in platforms if 'NVIDIA' in p.name)
devices = nvidia_platform.get_devices(device_type=cl.device_type.GPU)
context = cl.Context(devices)
queue = cl.CommandQueue(context)

# Validate work-group size
device = devices[0]
max_wg_size = device.max_work_group_size
max_dims = device.max_work_item_sizes
localsize = (32, 1)  # 32x1 work-group

if (localsize[0] * localsize[1] > max_wg_size) or \
   (localsize[0] > max_dims[0]) or (localsize[1] > max_dims[1]):
    raise ValueError(f"32x1 work-group not supported. Max: {max_wg_size} threads, {max_dims} dimensions")

# Create buffers
d_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_A)
d_b = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_B)
d_c = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_C.nbytes)

#--------------------------------------------------------------------
# Load and build kernel (C_elem_ij.cl)
#--------------------------------------------------------------------
kernel_code = open("C_elem_ij.cl").read()
program = cl.Program(context, kernel_code).build()
mmul = program.mmul
mmul.set_scalar_arg_dtypes([numpy.int32, None, None, None])

#--------------------------------------------------------------------
# Run kernel with 32x1 work-group
#--------------------------------------------------------------------
print(f"Testing 32x1 work-group on {device.name}...")
start = time()

for _ in range(COUNT):
    try:
        # Global size must be divisible by local size
        global_size = (N, N)
        mmul(queue, global_size, localsize, numpy.int32(N), d_a, d_b, d_c)
        cl.enqueue_copy(queue, h_C, d_c).wait()
    except cl.LogicError as e:
        print(f"Kernel failed: {e}")
        break

runtime = time() - start
gflops = (2 * N**3 * COUNT) / (runtime * 1e9)
print(f"GFLOPS: {gflops:.2f}")