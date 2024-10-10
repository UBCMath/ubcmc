# Design Decisions:
# Students should be able to ...
# * plot the graph of a function before learning about vectorization
# * work with matrices before learning about different numeric datatypes (int vs. float vs. complex)
# * manipulate matrices before learning about array dimensions

import numpy as np
import matplotlib.pyplot as plt

from numpy import sin, cos, exp, log, tan, sqrt, arcsin, arccos, arctan
from numpy import linspace
from scipy.linalg import solve
from matplotlib.pyplot import plot, scatter, grid, legend

def graph(f,a,b,n=100):
    "Plot the graph of a vectorized function f(x) over [a,b] using n points per unit."
    N = max(int((b - a)*n),100)
    x = linspace(a,b,N)
    y = f(x)
    plot(x,y)
    grid(True)

def point(x,y):
    scatter(x,y)
    
def matrix(A):
    B = np.matrix(A)
    if B.dtype == 'complex':
        return B
    return np.matrix(B,dtype=float,copy=True)

def identity(N):
    return matrix(np.identity(N))

def zeros(s):
    return matrix(np.zeros(s))

def ones(s):
    return matrix(np.ones(s))

def block(arrs):
    return matrix(np.block(arrs))

def diag(v):
    return matrix(np.diag(v))

# Elementary row operations

def add_row(A,c,i,j):
    "Add c times row i to row j"
    M = matrix(A)
    M[j,:] = c*M[i,:] + M[j,:]
    return M

def scale_row(A,c,i):
    "Multiply c times row i"
    M = matrix(A)
    M[i,:] = c*M[i,:]
    return M

def switch_row(A,i,j):
    "Switch row i and row j"
    M = matrix(A)
    ri = M[i,:].copy()
    rj = M[j,:].copy()
    M[i,:] = rj
    M[j,:] = ri
    return M

def rref(A):
    "Compute reduced row echelon form of matrix A."
    U = matrix(A)
    m,n = U.shape
    pivot = 0
    for j in range(0,n):
        imax = np.argmax(np.abs(U[pivot:,j]))
        if U[pivot + imax,j] != 0:
            U = switch_row(U,pivot,pivot + imax)
        else:
            continue
        U = scale_row(U,1/U[pivot,j],pivot)
        for i in range(m):
            if i == pivot:
                continue
            else:
                U = add_row(U,-U[i,j],pivot,i)
        pivot += 1
        U = np.round(U,14) + 0.0
        if pivot == m:
            break
    return U

# Vector operations

def dot(u,v):
    return np.sum(np.multiply(u,v))

# Visualization
def imshow(A):
    plt.imshow(A,cmap='RdBu')
    c = np.max(np.abs(A))
    plt.clim([-c,c])
    plt.colorbar()
    plt.show()