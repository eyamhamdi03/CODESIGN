#
# Matrix Multiplication Element By Element
# compute C=A*B
# C[i][j] for i: 0--> N-1, j! 0--> N-1
# The Program executes standard sequential processing using cpu

from helper import *
from definitions import *

import numpy

import pyopencl as cl

from time import time
from time import sleep

# A[N][N], B[N][N], C[N][N]
N = 512

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


print ("\n===== Sequential, matrix multiplication, order", N, "on host CPU ======\n")
h_C.fill(0.0)
start_time = time()
#print ("Skipping as this takes a long time to run!")
seq_mat_mul_sdot(N, h_A, h_B, h_C)
run_time = time() - start_time
gflops = (2 * N**3) / (run_time * 1e9)
print(f"GFLOPS: {gflops:.2f}")

print ("END OF 1 Sequential Matrix Multiplication",N,"*",N, "\n")

results(N, 1, run_time)