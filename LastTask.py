import numpy as np
import matplotlib.pyplot as plt

# Явная разностная схема

L = 0.02
T0 = 0
alpha = 0.001
q = 0

a = 0
b = a + L

nx = 10
t_final = 40

dx = L / nx
dt = 1 / t_final

x = np.linspace(a, b, nx)

dTdt = np.empty(nx)

t = np.arange(0, t_final, dt)


def explicitSol(T0, Tl, Tr, t, x, alpha, dt, dx, q):
    N = len(x)
    M = len(t)
    T = np.zeros((N,M))

    T[:, 0] = T0
    T[0, :] = Tl
    T[-1, :] = Tr
    for j in range(0, M-1):
        for i in range(1, N-1):
            T[i, j + 1] = T[i, j] + q*dt + dt * alpha**2/dx**2*(T[i+1, j] - 2*T[i, j] + T[i-1, j])
    return T


# Неявная разностная схема


def implicitSol(T0, Tl, Tr, t, x, alpha, dt, dx, q):
    N = len(x)
    M = len(t)

    T = np.zeros((N,M))

    T[:, 0] = T0
    T[0, :] = Tl
    T[-1, :] = Tr

    A = np.zeros((N-2, N-2))

    for i in range(0, N-2):
      A[i, i-1] = -alpha**2 * dt / dx**2
      A[i, i] = 1 + 2 * alpha**2 * dt / dx**2
      if i < N-3:
        A[i, i+1] = -alpha**2 * dt / dx**2
    # Time-stepping loop with implicit scheme
    for j in range(1, M):
      b = T[1:-1, j-1].copy()
      b[0] = b[0] + T[0, j] * alpha**2 * dt / dx**2
      b[-1] = b[-1] + T[-1, j] * alpha**2 * dt / dx**2
      b += q * dt
      solution = np.linalg.solve(A, b)
      T[1:-1, j] = solution
    return T


# Task 1
Tl = 20
Tr = 0
q = 0

res = explicitSol(T0, Tl, Tr, t, x, alpha, dt, dx, q)
plt.xlabel('x')
plt.ylabel('Temperature')
plt.title('Explicit Solution of Heat Equation with Implicit Scheme')
plt.plot(res)
plt.show()

res = implicitSol(T0, Tl, Tr, t, x, alpha, dt, dx, q)
plt.plot(x, res)
plt.xlabel('x')
plt.ylabel('Temperature')
plt.title('Implicit Solution of Heat Equation with Implicit Scheme')
plt.show()

# Task 2
Tl = np.linspace(20, 10, len(t))
Tr = np.linspace(0, -10, len(t))
q = 0

res = explicitSol(T0, Tl, Tr, t, x, alpha, dt, dx, q)
plt.xlabel('x')
plt.ylabel('Temperature')
plt.title('Explicit Range Solution of Heat Equation with Implicit Scheme')
plt.plot(res)
plt.show()

res = implicitSol(T0, Tl, Tr, t, x, alpha, dt, dx, q)
plt.plot(x, res)
plt.xlabel('x')
plt.ylabel('Temperature')
plt.title('Implicit Range Solution of Heat Equation with Implicit Scheme')
plt.show()

# Task 3

Tl = 293
Tr = 273
k = 0.025
ro = 1.2
c = 1005
q = 1000
q_new = q/ro/c
alpha_new = k/ro/c

res = explicitSol(T0, Tl, Tr, t, x, alpha_new, dt, dx, q_new)
plt.xlabel('x')
plt.ylabel('Temperature')
plt.title('Explicit Q Solution of Heat Equation with Implicit Scheme')
plt.plot(res)
plt.show()

res = implicitSol(T0, Tl, Tr, t, x, alpha_new, dt, dx, q_new)
plt.plot(x, res)
plt.xlabel('x')
plt.ylabel('Temperature')
plt.title('Implicit Q Solution of Heat Equation with Implicit Scheme')
plt.show()
