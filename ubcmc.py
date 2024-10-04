from numpy import sin, cos, exp, log, tan, sqrt, arcsin, arccos, arctan
from numpy import linspace
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