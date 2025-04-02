#
# Matrix Multiplication Element By Element
# compute C=A*B
# C[i][j] for i: 0--> N-1, j! 0--> N-1
# 1) Kernel version to execute multiplication with j=0, i=1

# 2) The Program asks for 2 inputs:
# localsize (4,8,16 or 32) --> Block Size = localsize*localsize

from helper import *
from definitions import *

import numpy as np

import pyopencl as cl

from time import time
from time import sleep

# A[N][N], B[N][N], C[N][N]
N = 8192

# Number of elements in the matrix
size = N * N
#true value
cval = float(N) * AVAL * BVAL

# A matrix
h_A = np.empty(size).astype(np.float32)
h_A.fill(AVAL)

# B matrix
h_B = np.empty(size).astype(np.float32)
h_B.fill(BVAL)

# C matrix
h_C = np.empty(size).astype(np.float32)

# =============================================
# OpenCL Setup (Force NVIDIA GPU)
# =============================================
# Get NVIDIA platform
platforms = cl.get_platforms()
nvidia_platform = next(p for p in platforms if 'NVIDIA' in p.name)
nvidia_devices = nvidia_platform.get_devices(device_type=cl.device_type.GPU)
context = cl.Context(nvidia_devices)
queue = cl.CommandQueue(context)

# Create buffers
d_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_A)
d_b = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_B)
d_c = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_C.nbytes)

# =============================================
# Kernel Configuration (32x1 Work-Group)
# =============================================
LOCALSIZE = (32, 1)  # Fixed work-group size
kernel_name = "C_elem_ij.cl"

# Validate work-group size
device = nvidia_devices[0]
if (LOCALSIZE[0] * LOCALSIZE[1] > device.max_work_group_size or
    LOCALSIZE[0] > device.max_work_item_sizes[0] or
    LOCALSIZE[1] > device.max_work_item_sizes[1]):
    raise ValueError(f"Device {device.name} doesn't support 32x1 work-groups")

# Load and build kernel
with open(kernel_name, 'r') as f:
    kernel_code = f.read()
program = cl.Program(context, kernel_code).build()
mmul = program.mmul
mmul.set_scalar_arg_dtypes([np.int32, None, None, None])

# =============================================
# Run Kernel
# =============================================
print(f"Running {COUNT} multiplications on {device.name} (Work-group: 32x1)")
start = time()

try:
    for _ in range(COUNT):
        mmul(
            queue, 
            (N, N),        # Global size
            LOCALSIZE,     # Local size (32x1)
            np.int32(N),   # Matrix dimension
            d_a, d_b, d_c  # Buffers
        )
    queue.finish()
except cl.Error as e:
    print(f"OpenCL error: {e}")

runtime = time() - start

# =============================================
# Results & Validation
# =============================================
cl.enqueue_copy(queue, h_C, d_c).wait()
gflops = (2 * N**3 * COUNT) / (runtime * 1e9)
print(f"\nPerformance: {gflops:.2f} GFLOPS")
print(f"Total runtime: {runtime:.2f} seconds")
print(f"Expected value: {cval}")
print(f"Actual value: {h_C[0]:.1f} (First element)")