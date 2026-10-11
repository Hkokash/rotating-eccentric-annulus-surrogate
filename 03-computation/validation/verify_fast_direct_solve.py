import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def solve_laplace_direct(r_i, r_o, d, n):
    L = r_o * 1.05
    x = np.linspace(-L, L, n)
    dx = x[1] - x[0]
    X, Y = np.meshgrid(x, x, indexing="ij")
    r_outer = np.sqrt(X**2 + Y**2)
    r_inner = np.sqrt(X**2 + (Y + d)**2)
    inside_outer = r_outer <= r_o
    inside_inner = r_inner <= r_i
    fluid = inside_outer & (~inside_inner)
    is_bc = inside_inner | (~inside_outer)

    idx = -np.ones((n, n), dtype=int)
    free = np.where(fluid)
    idx[free] = np.arange(len(free[0]))
    N = len(free[0])

    rows, cols, vals = [], [], []
    b = np.zeros(N)

    for k in range(N):
        i, j = free[0][k], free[1][k]
        rows.append(k); cols.append(k); vals.append(-4.0)
        for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            ii, jj = i + di, j + dj
            if is_bc[ii, jj]:
                bc_val = 1.0 if inside_inner[ii, jj] else 0.0
                b[k] -= bc_val
            else:
                rows.append(k); cols.append(idx[ii, jj]); vals.append(1.0)

    A = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    T_fluid = spla.spsolve(A, b)

    T = np.zeros((n, n))
    T[inside_inner] = 1.0
    T[free] = T_fluid

    # flux via Richardson-style two-radius sampling (bilinear interpolation)
    n_theta = 1440
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)

    def sample(xq, yq):
        ix = (xq - x[0]) / dx
        iy = (yq - x[0]) / dx
        ix0 = np.clip(ix.astype(int), 0, n - 2)
        iy0 = np.clip(iy.astype(int), 0, n - 2)
        fx, fy = ix - ix0, iy - iy0
        v00, v10 = T[ix0, iy0], T[ix0 + 1, iy0]
        v01, v11 = T[ix0, iy0 + 1], T[ix0 + 1, iy0 + 1]
        return v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy) + v01 * (1 - fx) * fy + v11 * fx * fy

    r1, r2 = r_i + 3 * dx, r_i + 6 * dx
    T1 = sample(r1 * np.cos(theta), -d + r1 * np.sin(theta))
    T2 = sample(r2 * np.cos(theta), -d + r2 * np.sin(theta))
    dTdn = (T2 - T1) / (r2 - r1)
    Q = -np.sum(dTdn) * ((r1 + r2) / 2) * (2 * np.pi / n_theta)
    return Q


def main():
    r_i, r_o, d = 0.10, 0.5, 0.0
    exact = 2 * np.pi / np.log(r_o / r_i)
    print("Exact (concentric, ln formula):", round(exact, 5))
    for n in [151, 301, 601, 901, 1201]:
        s = solve_laplace_direct(r_i, r_o, max(d, 1e-9), n)
        print(f"n={n:5d}  S_numerical={s:.5f}  %diff={100*abs(s-exact)/exact:.3f}%")


if __name__ == "__main__":
    main()
