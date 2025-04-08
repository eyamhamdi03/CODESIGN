# TP1partB.py - Modified to test multiple splits
from helper import *
from definitions import *
import numpy as np
import pyopencl as cl
from time import time

# Set N = 8192 (as per Part B)
N = 8192
COUNT = 1  # Number of repetitions

# Matrix initialization
AVAL = 1.0  # Value for matrix A
BVAL = 2.0  # Value for matrix B

# NVIDIA-only reference time (from previous calculation)
T_nvidia = 199.3850326538086   # Replace with your actual measurement

# Define various split configurations to test

split_configs = [
    [8192, 0, 0],        # Only dGPU (for verification)
    [4096, 2048, 2048],  # Original split
    [6144, 1024, 1024],  # More to dGPU
    [5120, 1536, 1536],  # Balanced heavy dGPU
    [3072, 3072, 2048],  # Balanced
]

best_speedup = 0
best_split = []
worst_speedup = float('inf')
worst_split = []

# Get all OpenCL devices once (outside the loop)
platforms = cl.get_platforms()
devices = []
for platform in platforms:
    devices.extend(platform.get_devices())

# Select devices (adjust indices based on your system)
selected_devices = [devices[0], devices[1], devices[2]]  # CPU, iGPU, dGPU

# Load kernel code once (outside the loop)
with open(r"C:/Users/eya-pc/OneDrive - Ministere de l'Enseignement Superieur et de la Recherche Scientifique/Bureau/CODESIGN/opencl_examples/C_elem_ij.cl") as f:
    kernel_source = f.read()

# Test all split configurations
for split in split_configs:
    if sum(split) != N:
        print(f"Skipping invalid split {split} (sum {sum(split)} != {N})")
        continue
    
    print(f"\n{'='*40}\nTesting split: {split}\n{'='*40}")
    
    try:
        # Re-initialize matrices for each run
        h_A = np.full((N, N), AVAL, dtype=np.float32)
        h_B = np.full((N, N), BVAL, dtype=np.float32)
        h_C = np.zeros((N, N), dtype=np.float32)
        
        split_offset = [0, split[0], split[0] + split[1]]
        
        # Create contexts and queues for current split
        contexts = []
        queues = []
        for device in selected_devices:
            ctx = cl.Context([device])
            queue = cl.CommandQueue(ctx)
            contexts.append(ctx)
            queues.append(queue)

        # Prepare buffers and kernels
        buffers = []
        kernels = []
        for i, (ctx, queue) in enumerate(zip(contexts, queues)):
            if split[i] == 0:
                continue  # Skip devices with 0 workload
                
            d_a = cl.Buffer(ctx, cl.mem_flags.READ_ONLY, size=split[i]*N*4)
            cl.enqueue_copy(queue, d_a, h_A, device_offset=split_offset[i]*N)
            mf = cl.mem_flags
            d_a = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A_part)
            d_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_B)
            d_c = cl.Buffer(ctx, mf.WRITE_ONLY, size=split[i] * N * 4)
            
            program = cl.Program(ctx, kernel_source).build()
            mmul = program.mmul
            mmul.set_scalar_arg_dtypes([np.int32, None, None, None])
            
            buffers.append((d_a, d_b, d_c))
            kernels.append(mmul)

        # Execute kernels
        localsize = 16
        global_sizes = [(split[i], N) for i in range(3) if split[i] > 0]
        
        start_time = time()
        for _ in range(COUNT):
            for i in range(len(kernels)):
                queue = queues[i]
                mmul = kernels[i]
                mmul(queue, global_sizes[i], (localsize, localsize), np.int32(N), *buffers[i])
            
            for queue in queues:
                queue.finish()
                
        total_time = time() - start_time
        if split == split_configs[0]:
            T_nvidia = total_time
        print(f"Split {split} - Time: {total_time:.2f}s")
        # Calculate speedup
        if total_time == 0:
            speedup = 0
        else:
            speedup = T_nvidia / total_time
        
        print(f"Split {split} - Time: {total_time:.2f}s - Speedup: {speedup:.2f}x")
        
        # Update best/worst
        if speedup > best_speedup:
            best_speedup = speedup
            best_split = split.copy()
            
        if speedup < worst_speedup:
            worst_speedup = speedup
            worst_split = split.copy()

    except Exception as e:
        print(f"Error with split {split}: {str(e)}")
        continue

# Final results
print("\n\n=== Final Results ===")
print(f"Best speedup: {best_speedup:.2f}x with split {best_split}")
print(f"Worst speedup: {worst_speedup:.2f}x with split {worst_split}")