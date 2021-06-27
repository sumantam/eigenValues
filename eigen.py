import numpy as np
import utils
import sys
import math

# Computes the largest magnitude eigenvalue of A via the power method, within
# a tolerance of delta or for iterations iterations
def power_method(A):
    x = np.zeros(A.shape[0])
    x[0] = 1.0
    while True:
        p = A @ x
        n = np.max(np.abs(p))
        x = (1.0 / n) * p
        yield n

# Computes all eigenvalues of a symmetric matrix A via the Jacobi method
def jacobi_method(A):
    norm = utils.outer_norm(A)
    yield norm
    while True:
        i, j = utils.outer_argmax(A)
        if i != 0 or j != 0:
            a = A.copy()
            theta = 0.5 * (A[i, i] - A[j, j]) / A[i, j]
            t = utils.sign(theta) / (abs(theta) + math.sqrt(theta**2 + 1))
            c = 1 / math.sqrt(t**2 + 1)
            s = c * t
            a[i, j] = a[j, i] = 0
            a[i, i] = A[i, i] + t * A[i, j]
            a[j, j] = A[j, j] - t * A[i, j]
            for l in range(A.shape[0]):
                if l != i and l != j:
                    a[i, l] = a[l, i] = c * A[i, l] + s * A[j, l]
                    a[j, l] = a[l, j] = c * A[j, l] - s * A[i, l]
            norm -= 2 * A[i, j]**2
            A[:] = a
            yield norm
        else:
            yield 0

def read_matrix(fd):
    x = [list(map(float, l.split())) for l in fd.readlines() if l.strip()]
    return np.array(x)

def main(args):
    A = read_matrix(sys.stdin)
    utils.validate_matrix(A)
    prev_eigenval = float("inf")
    if args.jacobi:
        for norm in jacobi_method(A):
            if norm < 0.0001:
                break
        for eigenval in np.diag(A):
            print("{:0.4f}".format(eigenval))
    else:
        for eigenval in power_method(A):
            if abs(eigenval - prev_eigenval) < 0.00001:
                break
            prev_eigenval = eigenval
        print("{:0.4f}".format(eigenval))

if __name__ == "__main__":
    import argparse






# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

'''
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
'''