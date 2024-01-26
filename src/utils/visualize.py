import os
import jax
import imageio
import numpy as np
import jax.numpy as jnp
from typing import Dict, Callable


def make_folder(name: str):
    if not (os.path.exists(name)):
        os.makedirs(name)


def ddim_sample_visual(
    params: dict,
    fn: Callable,
    pred_fn: Callable,
    x: jax.Array,
    time_pairs: jax.Array,
    key: jax.Array,
    var_params: Dict[str, jax.Array],
    steps: int = 250,
    eta: float = 0.0,  # control ddim sigma or stochasiticity
):
    def sample(i, vals):
        x, x_over_time, key = vals

        x_over_time = x_over_time.at[i].set(x)

        t_curr, t_prev = time_pairs[i][0], time_pairs[i][1]

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

        return x, x_over_time, key

    x_over_time = jnp.zeros([steps, *list(x.shape)])

    key, _ = jax.random.split(key)
    x, x_over_time, key = jax.lax.fori_loop(0, steps, sample, (x, x_over_time, key))
    return x, x_over_time


def ddpm_sample_visual(
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

    x_over_time = jnp.zeros([steps, *list(x.shape)])

    key, _ = jax.random.split(key)
    x, x_over_time, key = jax.lax.fori_loop(0, steps, sample, (x, x_over_time, key))
    return x, x_over_time


def create_gifs(
    x_over_time: np.asarray,  # Steps x Batch x H x W x C
    duration: float = 100,
    folder: str = "./example_gifs/",
    *,
    image_size: tuple = (256, 256),
    num_images: int = 10
):
    make_folder(folder)

    steps = x_over_time.shape[0]
    num_images = min(x_over_time.shape[1], num_images)

    for img_id in range(num_images):
        with imageio.get_writer(
            folder + "ex_{0}.gif".format(img_id), mode="I", duration=duration
        ) as writer:
            for step in range(steps):
                image = x_over_time[step][img_id].squeeze()
                image = jax.image.resize(image, image_size, "nearest")
                image = image * 255 + 0.5
                image = np.clip(image, 0, 255)
                writer.append_data(image.astype("uint8"))
