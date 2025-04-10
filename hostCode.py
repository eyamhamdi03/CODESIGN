# TP1partB_threaded.py - Parallel execution with multithreading and split
from helper import *
from definitions import *
import numpy as np
import pyopencl as cl
import threading
from time import time

N = 8192
COUNT = 10

AVAL = 1.0
BVAL = 2.0

split_configs = [
    [8192, 0, 0],         
    [7776, 144, 272],
    [7744, 160, 288],
    [7808, 128, 256],
]
best_speedup = 0
best_split = []
worst_speedup = float('inf')
worst_split = []

# Get OpenCL devices
platforms = cl.get_platforms()
devices = []
for platform in platforms:
    devices.extend(platform.get_devices())

selected_devices = [devices[0], devices[1], devices[2]]

# Load kernel
with open(r"C_elem_ij.cl") as f:
    kernel_source = f.read()

T_reference = None

for split in split_configs:
    if sum(split) != N:
        print(f"Skipping invalid split {split} (sum {sum(split)} != {N})")
        continue

    print(f"\n{'=' * 40}\nTesting split: {split}\n{'=' * 40}")

    try:
        h_A = np.full((N, N), AVAL, dtype=np.float32)
        h_B = np.full((N, N), BVAL, dtype=np.float32)
        h_C = np.zeros((N, N), dtype=np.float32)

        split_offset = [0, split[0], split[0] + split[1]]

        thread_times = [0.0, 0.0, 0.0]

        contexts, queues, buffers, kernels = [None]*3, [None]*3, [None]*3, [None]*3

        for i in range(3):
            if split[i] == 0:
                continue

            ctx = cl.Context([selected_devices[i]])
            queue = cl.CommandQueue(ctx)

            A_part = h_A[split_offset[i]:split_offset[i] + split[i], :].copy()
            mf = cl.mem_flags
            d_a = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A_part)
            d_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_B)
            d_c = cl.Buffer(ctx, mf.WRITE_ONLY, size=split[i] * N * 4)

            program = cl.Program(ctx, kernel_source).build()
            mmul = program.mmul
            mmul.set_scalar_arg_dtypes([np.int32, None, None, None])
            mmul.set_args(np.int32(N), d_a, d_b, d_c)

            contexts[i] = ctx
            queues[i] = queue
            buffers[i] = (d_a, d_b, d_c)
            kernels[i] = mmul

        def run_kernel(i):
            if split[i] == 0:
                return
            global_size = (split[i], N)
            local_size = (16, 16)
            start_t = time()
            kernels[i](queues[i], global_size, local_size, np.int32(N), *buffers[i])
            queues[i].finish()
            partial_C = np.empty((split[i], N), dtype=np.float32)
            cl.enqueue_copy(queues[i], partial_C, buffers[i][2]).wait()
            h_C[split_offset[i]:split_offset[i] + split[i], :] = partial_C
            thread_times[i] += time() - start_t

        # === Start COUNT timed executions ===
        start_total = time()
        for count_iter in range(COUNT):
            threads = []
            for i in range(3):
                if split[i] > 0:
                    t = threading.Thread(target=run_kernel, args=(i,))
                    threads.append(t)
                    t.start()

            for t in threads:
                t.join()
        total_time = time() - start_total

        if split == [8192, 0, 0]:
            T_reference = total_time

        speedup = T_reference / total_time if total_time and T_reference else 0

        print(f"\nTotal time over {COUNT} runs: {total_time:.4f} s - Speedup: {speedup:.2f}x")
        for i in range(3):
            if split[i] > 0:
                print(f"  Device {i} - {split[i]} rows - Total Time: {thread_times[i]:.4f} s")

        if speedup > best_speedup:
            best_speedup = speedup
            best_split = split.copy()
        if speedup < worst_speedup:
            worst_speedup = speedup
            worst_split = split.copy()

    except Exception as e:
        print(f"Error with split {split}: {str(e)}")
        continue

# Final summary
print("\n\n=== Final Results ===")
print(f"Best speedup: {best_speedup:.2f}x with split {best_split}")
print(f"Worst speedup: {worst_speedup:.2f}x with split {worst_split}")
