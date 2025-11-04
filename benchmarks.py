import numpy as np

# =========
# Objectives (return shape (-1,1); inputs X in [0,1]^d)
# =========

def branin_2d(X):
    X = np.atleast_2d(X)
    x1 = X[:, 0] * 15 - 5      # [-5, 10]
    x2 = X[:, 1] * 15          # [0, 15]
    a = 1.0
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)
    y = a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * np.cos(x1) + s
    return y.reshape(-1, 1)

def six_hump_camel_2d(X):
    """Domain: x∈[-3,3], y∈[-2,2] (global minima ~(-0.0898,0.7126) and (0.0898,-0.7126))."""
    X = np.atleast_2d(X)
    x = X[:, 0] * 6 - 3
    y = X[:, 1] * 4 - 2
    val = (4 - 2.1*x**2 + (x**4)/3)*x**2 + x*y + (-4 + 4*y**2)*y**2
    return val.reshape(-1, 1)

def goldstein_price_2d(X):
    """Domain: [-2,2]^2 (highly nonconvex)."""
    X = np.atleast_2d(X)
    x = X[:, 0] * 4 - 2
    y = X[:, 1] * 4 - 2
    a = 1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)
    b = 30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2)
    return (a * b).reshape(-1, 1)

def hartmann_3d(X):
    """Domain: [0,1]^3 (already normalized). Minimization benchmark."""
    X = np.atleast_2d(X)
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[3.0, 10.0, 30.0],
                  [0.1, 10.0, 35.0],
                  [3.0, 10.0, 30.0],
                  [0.1, 10.0, 35.0]])
    P = 1e-4 * np.array([[3689, 1170, 2673],
                         [4699, 4387, 7470],
                         [1091, 8732, 5547],
                         [381,  5743, 8828]])
    inner = np.sum(A[None,:,:] * (X[:,None,:] - P[None,:,:])**2, axis=2)
    y = -np.sum(alpha[None,:] * np.exp(-inner), axis=1)
    return y.reshape(-1,1)

def ackley_3d(X):
    """Domain: [-5,5]^3."""
    X = np.atleast_2d(X)
    z = X * 10 - 5
    a = 20.0; b = 0.2; c = 2*np.pi
    s1 = np.sqrt(np.mean(z**2, axis=1))
    s2 = np.mean(np.cos(c*z), axis=1)
    y = -a * np.exp(-b*s1) - np.exp(s2) + a + np.e
    return y.reshape(-1,1)

def rosenbrock_3d(X):
    """Domain: [-2,2]^3; global min at 0 (mapped from x=[1,1,1])."""
    X = np.atleast_2d(X)
    z = X * 4 - 2
    y = 0.0
    for i in range(2):  # pairs (0,1) and (1,2)
        y = y + 100*(z[:,i+1] - z[:,i]**2)**2 + (1 - z[:,i])**2
    return y.reshape(-1,1)

def levy_3d(X):
    """Domain: [-10,10]^3; global min 0 at all ones (mapped from 0.55)."""
    X = np.atleast_2d(X)
    z = X * 20 - 10
    w = 1 + (z - 1)/4
    term1 = np.sin(np.pi*w[:,0])**2
    term3 = (w[:,-1]-1)**2 * (1 + np.sin(2*np.pi*w[:,-1])**2)
    mid = np.sum((w[:, :-1]-1)**2 * (1 + 10*np.sin(np.pi*w[:, :-1]+1)**2), axis=1)
    y = term1 + mid + term3
    return y.reshape(-1,1)

# =========
# Constraints g(x) ≤ 0 are feasible; X in [0,1]^d
# =========

def constraint_ring_center(X, r=0.35):
    """Feasible inside a circle around (0.5,0.5)."""
    X = np.atleast_2d(X)
    g = (X[:,0]-0.5)**2 + (X[:,1]-0.5)**2 - r**2
    return g

def constraint_annulus(X, r0=0.35, width=0.10):
    """
    Feasible in an annulus (ring band) centered at (0.5,0.5).
    Implemented as TWO constraints to avoid abs():
      (1) (dist^2 - (r0 - width)^2) ≤ 0  -> outside inner circle is feasible
      (2) ((r0 + width)^2 - dist^2) ≤ 0  -> inside outer circle is feasible
    """
    X = np.atleast_2d(X)
    d2 = (X[:,0]-0.5)**2 + (X[:,1]-0.5)**2
    g1 = (r0 - width)**2 - d2     # ≤0 means d2 ≥ (r0 - w)^2
    g2 = d2 - (r0 + width)**2     # ≤0 means d2 ≤ (r0 + w)^2
    return [g1, g2]

def constraint_linear_wedge(X):
    """Feasible below a slanted plane: x1 + 0.8*x2 - 1.0 ≤ 0."""
    X = np.atleast_2d(X)
    return X[:,0] + 0.8*X[:,1] - 1.0

def constraint_wavy_band_2d(X):
    """
    Nonconvex; feasible below a sinusoid: sin(5π x1) + 0.5*x2 - 0.25 ≤ 0
    Creates multiple disjoint feasible stripes.
    """
    X = np.atleast_2d(X)
    return np.sin(5*np.pi*X[:,0]) + 0.5*X[:,1] - 0.25

def constraint_tilted_ellipse(X, a=0.35, b=0.15, theta=np.pi/6):
    """Feasible inside rotated ellipse centered at (0.55,0.55)."""
    X = np.atleast_2d(X)
    xc = X[:,0] - 0.55
    yc = X[:,1] - 0.55
    ct, st = np.cos(theta), np.sin(theta)
    u =  ct*xc + st*yc
    v = -st*xc + ct*yc
    g = (u/a)**2 + (v/b)**2 - 1.0
    return g

def constraint_3d_tunnel(X):
    """
    3D 'tunnel' feasibility: cylinder along x3 through the cube; adds difficulty.
    Feasible if radial distance to (0.5,0.5) in (x1,x2) ≤ 0.25.
    """
    X = np.atleast_2d(X)
    r2 = (X[:,0]-0.5)**2 + (X[:,1]-0.5)**2
    return r2 - 0.25**2

def constraint_3d_sine_shell(X):
    """
    Tough 3D nonconvex shell: |x3 - 0.5 - 0.15*sin(6π x1)*cos(6π x2)| ≤ 0.12
    Implemented via two inequalities.
    """
    X = np.atleast_2d(X)
    target = 0.5 + 0.15*np.sin(6*np.pi*X[:,0])*np.cos(6*np.pi*X[:,1])
    g_lower = (target - 0.12) - X[:,2]   # ≤0 means x3 ≥ target - 0.12
    g_upper = X[:,2] - (target + 0.12)   # ≤0 means x3 ≤ target + 0.12
    return [g_lower, g_upper]

# =========
# Ready-made benchmark sets (pick one)
# =========

# =========
# Ready-made benchmark sets (pick one)
# =========

BENCHMARKS = {
    # 2D classics
    "branin_easy_circle":       (branin_2d,          [constraint_ring_center]),
    "branin_wavy":              (branin_2d,          [constraint_wavy_band_2d]),
    "sixhump_wedge":            (six_hump_camel_2d,  [constraint_linear_wedge]),
    "goldstein_tiltedellipse":  (goldstein_price_2d, [constraint_tilted_ellipse]),

    # 2D ring band (two constraints)
    "goldstein_annulus":        (goldstein_price_2d, [lambda X: constraint_annulus(X)[0],
                                                      lambda X: constraint_annulus(X)[1]]),

    # 3D hard ones
    "hartmann3_tunnel":         (hartmann_3d,        [constraint_3d_tunnel]),
    "ackley3_shell":            (ackley_3d,          [lambda X: constraint_3d_sine_shell(X)[0],
                                                      lambda X: constraint_3d_sine_shell(X)[1]]),
    "rosenbrock3_tunnel":       (rosenbrock_3d,      [constraint_3d_tunnel]),
    "levy3_shell":              (levy_3d,            [lambda X: constraint_3d_sine_shell(X)[0],
                                                      lambda X: constraint_3d_sine_shell(X)[1]]),
}


# Small helper to normalize constraints that may return a list
def expand_constraints(constraints):
    normed = []
    for con in constraints:
        # some entries in BENCHMARKS are already expanded lists produced by *calling* a factory
        if callable(con):
            normed.append(con)
        else:
            # a precomputed list of arrays/functions
            for g in con:
                if callable(g):
                    normed.append(g)
                else:
                    raise ValueError("Constraint list should contain callables.")
    return normed

