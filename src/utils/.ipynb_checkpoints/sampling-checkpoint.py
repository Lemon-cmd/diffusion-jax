import jax
from math import pi
import numpy as np
import jax.numpy as jnp
from typing import Callable, Dict


def get_time_pairs(timesteps: int = 1000, sampling_steps: int = 250) -> np.ndarray:
    times = np.linspace(-1, timesteps - 1, sampling_steps + 1).astype("int16")
    times = list(reversed(times.tolist()))
    time_pairs = list(zip(times[:-1], times[1:]))
    return np.asarray(time_pairs)


def get_var_params(betas: np.ndarray) -> Dict[str, jax.Array]:
    alphas = 1.0 - betas
    alphas_cp = jnp.asarray(np.cumprod(alphas))
    alphas_cp_recip = 1.0 / alphas_cp
    sqrt_alphas_cp = jnp.sqrt(alphas_cp)
    sqrt_alphas_cp_recip = alphas_cp_recip**0.5
    sqrt_alphas_cp_recip_minus_one = (alphas_cp_recip - 1.0) ** 0.5

    sqrt_one_minus_alphas_cp = jnp.sqrt(1.0 - alphas_cp)

    alphas = jnp.asarray(alphas)
    alphas_cp_prev = jnp.concatenate([jnp.ones(1), alphas_cp[:-1]])
    posterior_var = betas * (1.0 - alphas_cp_prev) / (1.0 - alphas_cp)

    posterior_log_var_clamped = jnp.log(
        jnp.clip(posterior_var, a_min=1e-20, a_max=None)
    )
    posterior_mu_coeff1 = betas * jnp.sqrt(alphas_cp_prev) / (1.0 - alphas_cp)
    posterior_mu_coeff2 = (1.0 - alphas_cp_prev) * jnp.sqrt(alphas) / (1.0 - alphas_cp)

    return {
        "betas": betas,
        "alphas": alphas[:, None, None, None],
        "alphas_cp": alphas_cp[:, None, None, None],
        "sqrt_alphas_cp": sqrt_alphas_cp[:, None, None, None],
        "sqrt_alphas_cp_recip": sqrt_alphas_cp_recip[:, None, None, None],
        "sqrt_alphas_cp_recip_minus_one": sqrt_alphas_cp_recip_minus_one[
            :, None, None, None
        ],
        "sqrt_one_minus_alphas_cp": sqrt_one_minus_alphas_cp[:, None, None, None],
        "posterior_var": posterior_var[:, None, None, None],
        "posterior_log_var": posterior_log_var_clamped[:, None, None, None],
        "posterior_mu_coeff1": posterior_mu_coeff1[:, None, None, None],
        "posterior_mu_coeff2": posterior_mu_coeff2[:, None, None, None],
    }


def linear_schedule(
    steps: int = 1000, *, start: float = 1e-4, end: float = 0.02
) -> np.ndarray:
    return np.linspace(start, end, steps)


def cosine_schedule(steps: int = 1000, *, s: float = 0.0008) -> np.ndarray:
    t = jnp.linspace(0, steps, steps + 1) / steps

    ft = jnp.cos((t + s) / (1 + s) * 0.5 * pi) ** 2
    alphas_cumprod = ft / ft[0]

    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(np.asarray(betas), 0.0, 0.999)


def sigmoid_schedule(
    steps: int = 1000,
    *,
    start: float = -3,
    end: float = 3,
    tau: float = 1,
    min_val: float = 1e-5
) -> np.ndarray:
    t = jnp.linspace(0, steps, steps + 1) / steps

    v_start, v_end = map(lambda z: jax.nn.sigmoid(z / tau), (start, end))

    ft = (-jax.nn.sigmoid((t * (end - start) + start) / tau) + v_end) / (
        v_end - v_start
    )
    alphas_cumprod = ft / ft[0]

    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(np.asarray(betas), min_val, 0.999)


def predict_x0_from_noise(
    noise: jax.Array,
    x_t: jax.Array,
    t: jax.Array,
    var_params: Dict[str, jax.Array],
):
    a = var_params["sqrt_alphas_cp_recip"][t]  # 1.0 / sqrt(alpha_bar_t)
    b = var_params["sqrt_alphas_cp_recip_minus_one"][t]  # sqrt(1.0 / alpha_bar_t - 1.0)
    return a * x_t - b * noise


def predict_noise_from_x0(
    x0: jax.Array,
    x_t: jax.Array,
    t: jax.Array,
    var_params: Dict[str, jax.Array],
):
    a = var_params["sqrt_alphas_cp_recip"][t]  # 1.0 / sqrt(alpha_bar_t)
    b = var_params["sqrt_alphas_cp_recip_minus_one"][t]  # sqrt(1.0 / alpha_bar_t - 1.0)
    return (a * x_t - x0) / b


def predict_v_from_x0(
    x0: jax.Array,
    noise: jax.Array,
    t: jax.Array,
    var_params: Dict[str, jax.Array],
):
    a = var_params["sqrt_alphas_cp"][t]
    b = var_params["sqrt_one_minus_alphas_cp"][t]
    return a * noise - b * x0


def predict_x0_from_v(
    v: jax.Array,
    x_t: jax.Array,
    t: jax.Array,
    var_params: Dict[str, jax.Array],
):
    a = var_params["sqrt_alphas_cp"][t]
    b = var_params["sqrt_one_minus_alphas_cp"][t]
    return a * x_t - b * v


def model_pred_v(
    params: dict,
    fn: Callable,
    x_t: jax.Array,
    t: jax.Array,
    var_params: Dict[str, jax.Array],
):
    v = fn({"params": params}, x_t, t[None].repeat(x_t.shape[0]))
    x0 = predict_x0_from_v(v, x_t, t, var_params)

    x0 = jax.lax.clamp(-1.0, x0, 1.0)

    noise = predict_noise_from_x0(x0, x_t, t, var_params)
    return x0, noise


def model_pred_x0(
    params: dict,
    fn: Callable,
    x_t: jax.Array,
    t: jax.Array,
    var_params: Dict[str, jax.Array],
):
    x0 = fn({"params": params}, x_t, t[None].repeat(x_t.shape[0]))

    x0 = jax.lax.clamp(-1.0, x0, 1.0)

    noise = predict_noise_from_x0(x0, x_t, t, var_params)
    return x0, noise


def model_pred_noise(
    params: dict,
    fn: Callable,
    x_t: jax.Array,
    t: jax.Array,
    var_params: Dict[str, jax.Array],
):
    noise = fn({"params": params}, x_t, t[None].repeat(x_t.shape[0]))

    x0 = predict_x0_from_noise(noise, x_t, t, var_params)

    x0 = jax.lax.clamp(-1.0, x0, 1.0)
    return x0, noise


def ddpm_sample(
    params: dict,
    fn: Callable,
    pred_fn: Callable,
    x: jax.Array,
    times: jax.Array,
    key: jax.Array,
    var_params: Dict[str, jax.Array],
    steps: int = 1000,
):
    times = jnp.flip(times)

    def sample(i, vals):
        x, key = vals

        t = times[i]
        x0, _ = pred_fn(params, fn, x, t, var_params)
        key, _ = jax.random.split(key)

        # compute mean
        mean = (
            var_params["posterior_mean_coeff1"][t] * x0
            + var_params["posterior_mean_coeff2"][t] * x
        )

        x = mean
        x = jnp.where(
            i > 0,
            x
            + jnp.exp(0.5 * var_params["posterior_log_var"][t])
            * jax.random.normal(key, shape=x.shape),
            x,
        )
        return x, key

    key, _ = jax.random.split(key)
    x, key = jax.lax.fori_loop(0, steps, sample, (x, key))
    return x


def ddim_sample(
    params: dict,
    fn: Callable,
    pred_fn: Callable,
    x: jax.Array,
    time_pairs: jax.Array,
    key: jax.Array,
    var_params: Dict[str, jax.Array],
    steps: int = 250,
    eta: float = 0.0,  # control ddim sigma and stochasticity
):
    def sample(i, vals):
        x, key = vals

        time_pair = time_pairs[i]

        t_curr = time_pair[0]
        t_prev = time_pair[1]

        x0, xt_dir = pred_fn(params, fn, x, t_curr, var_params)

        alpha_cp_curr = var_params["alphas_cp"][t_curr]
        alpha_cp_prev = var_params["alphas_cp"][t_prev]

        sigma = eta
        sigma *= pow((1.0 - alpha_cp_prev) / (1.0 - alpha_cp_curr), 0.5)
        sigma *= pow(1.0 - alpha_cp_curr / alpha_cp_prev, 0.5)

        key, _ = jax.random.split(key)

        x = jnp.where(
            t_prev >= 0,
            (
                (alpha_cp_prev**0.5) * x0
                + pow(1.0 - alpha_cp_prev - pow(sigma, 2.0), 0.5) * xt_dir
                + jax.random.normal(key, shape=x.shape) * sigma
            ),
            x0,
        )

        return x, key

    key, _ = jax.random.split(key)
    x, key = jax.lax.fori_loop(0, steps, sample, (x, key))
    return x
