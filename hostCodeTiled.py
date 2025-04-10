from helper import *
from definitions import *
import numpy as np
import pyopencl as cl
import threading
from time import time

N = 8192
TILE = 256  # tile size (TILE x TILE)
COUNT = 1
AVAL = 1.0
BVAL = 2.0

device_weights = [0.6, 0.25, 0.15]
num_tiles_per_dim = N // TILE
total_tiles = num_tiles_per_dim ** 2

# Assign tiles to devices based on weights
tile_assignments = [[] for _ in range(3)]
tile_indices = [(i, j) for i in range(num_tiles_per_dim) for j in range(num_tiles_per_dim)]

for idx, (i, j) in enumerate(tile_indices):
    device_id = int((idx / total_tiles) / (1 / len(device_weights)))  # simple round-robin based on weight
    tile_assignments[device_id % 3].append((i, j))  # simple wrap-around

# Get devices
platforms = cl.get_platforms()
devices = []
for platform in platforms:
    devices.extend(platform.get_devices())
selected_devices = [devices[0], devices[1], devices[2]]

# Load kernel
with open("C_elem_ij.cl") as f:
    kernel_source = f.read()

h_A = np.full((N, N), AVAL, dtype=np.float32)
h_B = np.full((N, N), BVAL, dtype=np.float32)
h_C = np.zeros((N, N), dtype=np.float32)

thread_times = [0.0, 0.0, 0.0]

contexts, queues, programs, kernels = [], [], [], []

# Setup OpenCL
for i in range(3):
    ctx = cl.Context([selected_devices[i]])
    queue = cl.CommandQueue(ctx)
    program = cl.Program(ctx, kernel_source).build()
    kernel = program.mmul
    kernel.set_scalar_arg_dtypes([np.int32, None, None, None])
    contexts.append(ctx)
    queues.append(queue)
    programs.append(program)
    kernels.append(kernel)

# Pre-copy B to device once
device_buffers = []
for i in range(3):
    mf = cl.mem_flags
    d_b = cl.Buffer(contexts[i], mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_B)
    device_buffers.append({'B': d_b})

def run_tiles(i):
    ctx = contexts[i]
    queue = queues[i]
    kernel = kernels[i]
    d_b = device_buffers[i]['B']
    mf = cl.mem_flags

    start = time()
    for tile_i, tile_j in tile_assignments[i]:
        # Get A sub-block
        A_tile = h_A[tile_i*TILE:(tile_i+1)*TILE, :].copy()
        d_a = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A_tile)

        # Output sub-block
        d_c = cl.Buffer(ctx, mf.WRITE_ONLY, size=TILE * TILE * 4)

        kernel.set_args(np.int32(N), d_a, d_b, d_c)

        global_size = (TILE, TILE)
        local_size = (16, 16)

        kernel(queue, global_size, local_size)
        queue.finish()

        # Copy back result
        partial_C = np.empty((TILE, TILE), dtype=np.float32)
        cl.enqueue_copy(queue, partial_C, d_c).wait()
        h_C[tile_i*TILE:(tile_i+1)*TILE, tile_j*TILE:(tile_j+1)*TILE] = partial_C

    thread_times[i] = time() - start

# === Run tiled kernels in parallel ===
start_total = time()
threads = []
for i in range(3):
    t = threading.Thread(target=run_tiles, args=(i,))
    threads.append(t)
    t.start()
for t in threads:
    t.join()
total_time = time() - start_total

# === Print results ===
print(f"\nTotal time: {total_time:.4f} s")
for i in range(3):
    print(f"  Device {i} - {len(tile_assignments[i])} tiles - Time: {thread_times[i]:.4f} s")
