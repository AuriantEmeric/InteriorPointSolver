"""
Barrier method implementation and various plots to analyse convergence.
In some rare cases, the convergence fails. The asserts are here to avoid an infinite loop.
"""

import numpy as np

def centering_step(Q, p, A, b, t, v0, eps, max_iter=100, eps_=1e-10):
    """Centering step using Newton optimization."""
    v_list = [v0]
    v = v0.copy()
    iter = 0

    while True:
        # Get newton step
        dx_nt, lbd2, grad = newton_step(Q, p, A, b, v, t, eps_=eps_)

        # Backtracking line search
        t_step = line_search(Q, p, A, b, v, t, dx_nt, grad, eps_=eps_)

        # Update and store
        v = v + t_step * dx_nt
        v_list.append(v.copy())

        # End condition
        if lbd2 / 2 <= eps:
            return v_list

        # Avoid an infinite loop
        assert iter < max_iter, "Centering step failed, too many iterations"
        iter += 1


def newton_step(Q, p, A, b, v, t, eps_=1e-10):
    """Compute required values for Newton optimization."""
    # Compute grad
    grad_barrier = A.T / (np.clip(b - A @ v, a_min=0, a_max=None) + eps_)[None, ...]
    grad = t * (2 * Q @ v + p) + np.sum(grad_barrier, axis=-1)

    # Compute hessian
    denom = np.power(np.clip(b - A @ v, a_min=0, a_max=None), 2) + eps_
    hess_barrier = np.einsum('ij, il -> ijl', A, A) / denom[..., None, None]
    hess = t * 2 * Q + np.sum(hess_barrier, axis=0)
    hess_inv = np.linalg.inv(hess)

    # Newton step and lambda
    dx_nt = - hess_inv @ grad
    lbd2 = grad.T @ hess_inv @ grad
    return dx_nt, lbd2, grad


def fct(Q, p, A, b, v, t, eps_=1e-10):
    """Return t * f + phi."""
    phi = np.sum(np.log(np.clip(b - A @ v, a_min=0, a_max=None) + eps_))
    return t * (v.T @ Q @ v + p.T @ v) - phi


def quadratic_fct(Q, p, v):
    """Return the value of the quadratic function in v."""
    return v.T @ Q @ v + p.T @ v


def line_search(Q, p, A, b, v, t, dx_nt, grad, alpha=0.1, beta=0.8, max_iter=100, eps_=1e-10):
    """Find the parameter t for the Newton step using backtracking."""
    t_step = 1
    dir = alpha * grad.T @ dx_nt
    fx = fct(Q, p, A, b, v, t, eps_=eps_)
    iter = 0

    while True:
        # End condition
        fx_dx = fct(Q, p, A, b, v + t_step * dx_nt, t, eps_=eps_)
        if fx_dx <= fx + t_step * dir or iter > max_iter:
            return t_step

        # Update
        t_step = beta * t_step

        # Avoid infinite loop
        iter += 1


def barr_method(Q, p, A, b, v0, eps, t, mu, max_iter=100, eps_=1e-10):
    """Barrier method. Return a list of optimal v for increasing t."""
    v_list = [v0]
    v = v0.copy()
    m = A.shape[0]
    iter = 0

    while True:
        # Centering step
        v_list.append(centering_step(Q, p, A, b, t, v, eps, max_iter=max_iter, eps_=eps_)[-1])

        # Update
        v = v_list[-1].copy()

        # Stopping criterion
        if m / t < eps:
            return v_list

        # Increase
        t = mu * t

        assert iter < max_iter, "Barrier method failed, too many iterations"
        iter += 1


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    # Parameters
    n = 20
    d = 15
    lmbd = 10
    eps = 1e-5
    mu_list = [2, 15, 50, 100, 150]
    t = 1

    # Store optimal values
    v_opt = []
    f_min = []

    # Get random data
    X = np.random.normal(size=(n, d))
    y = np.random.normal(size=n)

    # Create quadratic problem
    A = np.concatenate((X.T, -X.T), axis=0)
    b = lmbd * np.ones(2 * d)
    Q = 0.5 * np.identity(n)
    p = -y

    # Duality gap during barrier method
    plt.figure()
    for i in tqdm(range(len(mu_list))):
        v0 = np.zeros(n)
        mu = mu_list[i]

        v_list = barr_method(Q, p, A, b, v0, eps, t, mu)
        f_list = [quadratic_fct(Q, p, v) for v in v_list]

        plt.step(np.arange(len(v_list)), f_list - f_list[-1], label=f'mu = {mu}')
    plt.legend()
    plt.semilogy()
    plt.title("Evolution of the duality gap for different mu")
    plt.ylabel("Estimated duality gap")
    plt.xlabel("Centering step")
    plt.show()

    # Duality gap during barrier method
    mu_list = np.concatenate((np.array([2, 3, 4]), np.arange(5, 205, 5)))
    v_opt = []
    f_min = []
    n_step = []

    plt.figure()
    for i in tqdm(range(len(mu_list))):
        v0 = np.zeros(n)
        mu = mu_list[i]

        v_list = barr_method(Q, p, A, b, v0, eps, t, mu)
        v_opt.append(v_list[-1])
        f_min.append(quadratic_fct(Q, p, v_opt[-1]))
        n_step.append(len(v_list))

    plt.title("Duality gap obtained with different mu")
    plt.ylabel("Estimated duality gap")
    plt.xlabel("mu")
    plt.plot(mu_list, f_min - np.min(f_min))
    plt.show()

    # Compute an estimation of w
    w_opt = [np.linalg.lstsq(X, y + v, rcond=None)[0] for v in v_opt]
    w_min = w_opt[np.argmin(f_min)]
    norm = [np.linalg.norm(w - w_min) for w in w_opt]

    plt.figure()
    plt.title("Norm of the difference of w obtained with different mu")
    plt.ylabel("w_mu - w*")
    plt.xlabel("mu")
    plt.plot(mu_list, norm)
    plt.show()

    # Compare number of steps to achieve required precision
    plt.figure()
    plt.title("Number of steps for various mu")
    plt.xlabel("mu")
    plt.ylabel("Number of steps")
    plt.plot(mu_list, n_step)
    plt.show()
