import jax
from jax import numpy as jnp, random as jr, tree_util as jtu, vmap, lax
from jax.numpy import linalg as jla
from numpy.polynomial.hermite_e import herme2poly
from typing import Callable

factorial = lambda k: jnp.prod(jnp.arange(k, 0, -1).astype(float))
He = lambda k, x: jnp.polyval(herme2poly([0] * k + [1])[::-1], x)
sigma = jax.nn.relu


# Utility function that iteratively computes the mean of a function f over a dataset
# Useful for computing averages over ghost batches
def laxmean(f, data):
    n = len(jtu.tree_leaves(data)[0])
    x0 = jtu.tree_map(lambda x: x[0], data)
    out_tree = jax.eval_shape(f, x0)
    avg_init = jtu.tree_map(
        lambda leaf: jnp.zeros(leaf.shape, dtype=leaf.dtype), out_tree
    )

    def step(avg, x):
        avg = jtu.tree_map(lambda a, b: a + b / n, avg, f(x))
        return avg, None

    return lax.scan(step, avg_init, data)[0]


def experiment(
    fstar: Callable,  # target function (should map \R \to \R)
    key: jax.Array,  # random key
    n: int,  # number of data points
    d: int,  # input dimension
    m: int,  # network width
    sigma2: float,  # noise variance
    test_set_size: int = 10000,  # size of the test set
    val_set_size: int = 10000,  # size of the validation set
    batch_size=-1,  # ghost batch size (useful for large n)
):
    target_key, model_key, data_key, val_key, test_key = jr.split(key, 5)
    wstar = jr.normal(target_key, (d,))
    wstar = wstar / jla.norm(wstar)

    def gen_batch(key, batch_size):
        x_key, y_key = jr.split(key)
        x = jr.normal(x_key, (batch_size, d))
        y = fstar(x @ wstar) + jnp.sqrt(sigma2) * jr.normal(y_key, (batch_size,))
        return x, y

    def model_init(key):
        w_key, b_key, a_key = jr.split(key, 3)
        w = jr.normal(w_key, (m, d)) / jnp.sqrt(d)
        b = jr.normal(b_key, (m,))
        a = jr.rademacher(a_key, (m,))
        # Symmetrize
        w = w.at[m // 2 :, :].set(w[: m // 2, :])
        b = b.at[m // 2 :].set(b[: m // 2])
        a = a.at[m // 2 :].set(-a[: m // 2])
        return w, b, a

    w, b, a = model_init(model_key)

    # Split the computations over batches of size batch_size
    # We save the random seed for each batch to reuse samples
    if batch_size < 0:
        batch_size = n
    n_batch = n // batch_size
    assert n_batch * batch_size == n  # batch_size should divide n
    batch_keys = jr.split(data_key, n_batch)

    def get_C0_C1(batch_key):
        x, y = gen_batch(batch_key, batch_size)
        return y.mean(), y @ x / batch_size

    C0, C1 = laxmean(get_C0_C1, batch_keys)

    # Loss Function
    loss = lambda w, x, y: jnp.mean((sigma(x @ w.T + b) @ a - y) ** 2)

    # Gradient at Initialization
    def w_grad_fn(batch_key):
        x, y = gen_batch(batch_key, batch_size)
        y = y - C0 - x @ C1  # remove the linear part
        return jax.grad(loss)(w, x, y)

    dw = laxmean(w_grad_fn, batch_keys)
    avg_norm = jnp.sqrt(jnp.sum(dw**2, 1).mean(0))
    w = -dw / avg_norm  # one step of GD

    # If H \in \R^{n \times m} is the random feature matrix,
    # we want to minimize \frac{1}{n}\|H @ a - y\|^2 + \lambda \|a\|^2/2 which has closed form
    # a = (H^T H/n + \lambda I)^{-1} (H^T y)/n
    # Both H^T H/n and H^t y/n are averages over the data points and can be computed via sequential averaging:

    def K_fn(key):
        x, y = gen_batch(key, batch_size)
        y = y - C0 - x @ C1  # remove the linear part
        H = sigma(x @ w.T + b)  # random feature
        return H.T @ H / batch_size, H.T @ y / batch_size

    HtH, Hty = laxmean(K_fn, batch_keys)

    # reuse the SVD computation for all regularization parameters
    U, S, Vt = jla.svd(HtH)
    UtHty = U.T @ Hty
    a_fn = lambda wd: Vt.T @ (UtHty / (S + wd))

    # Generate a validation set to tune the regularization parameter
    val_x, val_y = gen_batch(val_key, val_set_size)
    val_y = val_y - C0 - val_x @ C1  # remove the linear part

    # Grid search over the regularization parameter
    wds = jnp.geomspace(1e-10, 1e15, 100)
    val_losses = vmap(
        lambda wd: jnp.mean((sigma(val_x @ w.T + b) @ a_fn(wd) - val_y) ** 2)
    )(wds)
    a = a_fn(wds[jnp.argmin(val_losses)])

    # Compute the Test Loss
    test_x, test_y = gen_batch(test_key, test_set_size)
    test_y = test_y - C0 - test_x @ C1
    test_loss = jnp.mean((sigma(test_x @ w.T + b) @ a - test_y) ** 2) - sigma2

    return test_loss


fstar = lambda x: He(2, x) / jnp.sqrt(factorial(2))
key = jr.PRNGKey(seed=0)
print("Running Experiment (expected value is ~0.096)")
print("Test Loss: ", experiment(fstar, key, n=10000, d=100, m=1000, sigma2=1))
