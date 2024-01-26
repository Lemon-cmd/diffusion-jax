import tqdm
import optax
import torch
import torchvision
import jax, flax
import numpy as np
import equinox as eqx
import jax.numpy as jnp
import flax.linen as nn
from functools import partial
from flax.core.frozen_dict import FrozenDict

from .misc import *

from .sampling import (
    model_pred_v,
    model_pred_x0,
    model_pred_noise,
    ddim_sample,
    get_time_pairs,
)

from .data import DiffusionLoader
from typing import Callable, Optional

REPLICATE, UNREPLICATE = flax.jax_utils.replicate, flax.jax_utils.unreplicate


def normalize_to_neg_one_to_one(t):
    # assuming t is between 0 and 1
    return 2 * t - 1


def unnormalize_to_zero_to_one(t):
    # assuming t is between -1 and 1
    return (t + 1) * 0.5


def get_pred_fn(loss_type: str = "pred_x0") -> Callable:
    if loss_type == "pred_x0":
        return model_pred_x0
    elif loss_type == "pred_v":
        return model_pred_v
    elif loss_type == "pred_noise":
        return model_pred_noise
    else:
        raise Exception("Invalid Loss Type.")


def ema_update(
    params: FrozenDict,
    ema_params: FrozenDict,
    steps: int,
    max_ema_decay: float = 0.999,
    min_ema_decay: float = 0.5,
    ema_decay_power: float = 0.6666666,
    ema_inv_gamma: float = 1.0,
    start_ema_update_after: int = 100,
    update_ema_every: int = 10,
) -> FrozenDict:
    """Incorporates updated model parameters into an exponential moving averaged
    version of a model. It should be called after each optimizer step."""

    def calculate_decay():
        decay = 1.0 - (1.0 + (steps / ema_inv_gamma)) ** (-ema_decay_power)
        return np.clip(decay, min_ema_decay, max_ema_decay)

    if steps < start_ema_update_after:
        """When EMA is not updated, return the current params"""
        return params

    if steps % update_ema_every == 0:
        decay = calculate_decay()
        decay_avg = 1.0 - decay

        return jax.tree_util.tree_map(
            lambda ema, p_new: decay_avg * ema + decay * p_new,
            ema_params,
            params,
        )

    return ema_params


def objective(
    params: FrozenDict,
    fn: Callable,
    x: jax.Array,
    y: jax.Array,
    t: jax.Array,
    loss_weight: jax.Array,
) -> jax.Array:
    """Mean Square Error With Scaling"""

    def __last_shape(z):
        """Get Shape Excluding Batch Dim."""
        return [i for i in range(1, len(z.shape))]

    yh = fn({"params": params}, x, t)
    return jnp.mean(jnp.square(yh - y).mean(__last_shape(y)) * loss_weight)


@partial(eqx.filter_pmap, in_axes=(None, 0, None, 0, 0, 0, 0, 0), axis_name="batch")
def update(
    model: nn.Module,
    params: FrozenDict,
    opt,
    opt_state: FrozenDict,
    x: jax.Array,
    y: jax.Array,
    t: jax.Array,
    loss_weight: jax.Array,
):
    losses, grads = jax.value_and_grad(objective)(
        params, model.apply, x, y, t, loss_weight
    )
    losses, grads = map(lambda v: jax.lax.pmean(v, "batch"), (losses, grads))
    updates, opt_state = opt.update(grads, opt_state, params=params)
    params = optax.apply_updates(params, updates)

    return (params, opt_state, losses)


def train_model(
    model: nn.Module,
    params: FrozenDict,
    ema_params: FrozenDict,
    opt,
    opt_state: FrozenDict,
    train_loader: DiffusionLoader,
    config: dict,
):
    key = config["key"]
    if key is not None:
        key = jax.random.PRNGKey(config["timesteps"])
        key, _ = jax.random.split(key)

    train_status = tqdm.tqdm(range(1, config["epochs"] + 1))

    pred_fn = get_pred_fn(train_loader.loss_type)
    sampling = eqx.filter_pmap(
        ddim_sample, in_axes=(0, None, None, 0, None, None, None, None, None)
    )
    time_pairs = get_time_pairs(config["timesteps"], config["sampling_steps"])

    image_folder = config["result_path"] + "/images"
    ckpts_folder = config["result_path"] + "/ckpts"

    ckpt_manager = create_checkpoint_manager(
        ckpts_folder, max_to_keep=config["max_to_keep"]
    )
    make_folder(image_folder)

    step = ckpt_manager.latest_step()
    if step is not None:
        ckpt = restore_model(
            ckpt_manager,
            target={
                "params": params,
                "opt_state": opt_state,
                "ema_params": ema_params,
                "config": config,
            },
        )
        params, opt_state, ema_params, config = (
            ckpt["params"],
            ckpt["opt_state"],
            ckpt["ema_params"],
            ckpt["config"],
        )
    else:
        step = 0

    params, ema_params, opt_state = map(REPLICATE, (params, ema_params, opt_state))

    iteration = config["iteration"]
    for epoch in train_status:
        step = step + 1
        running_loss = 0.0

        for _ in range(len(train_loader)):
            # grab data from data loader
            x_t, labels, t, loss_weight = train_loader(config["timesteps"])

            # split to device
            x_t, labels, t, loss_weight = map(shard, (x_t, labels, t, loss_weight))

            # update model parameters
            params, opt_state, losses = update(
                model, params, opt, opt_state, x_t, labels, t, loss_weight
            )

            # compute ema
            ema_params = ema_update(
                params,
                ema_params,
                iteration,
                config["max_ema_decay"],
                config["min_ema_decay"],
                config["ema_decay_power"],
                config["ema_inv_gamma"],
                config["start_ema_update_after"],
                config["update_ema_every"],
            )

            # accummulate loss
            running_loss += losses[0].item()
            iteration += 1

        # average loss
        running_loss /= len(train_loader)

        train_status.set_description(
            "Epoch: {0}    Loss: {1:.6f}".format(epoch, running_loss)
        )

        if epoch % config["save_every_k"] == 0:
            """Visualize Sampling Quality
            Note--- we evaluate with ema_params
            """

            key, _ = jax.random.split(key)
            x_t = np.random.normal(0, 1, size=[128, *list(x_t.shape[2:])])

            images = unshard(
                np.array(
                    sampling(
                        ema_params,
                        model.apply,
                        pred_fn,
                        shard(x_t),
                        time_pairs,
                        key,
                        config["var_params"],
                        config["sampling_steps"],
                        config["eta"],
                    )
                )
            )

            images = np.concatenate([x_t, images], axis=0).transpose(0, 3, 1, 2)
            images = torch.from_numpy(images)
            images = unnormalize_to_zero_to_one(images)

            torchvision.utils.save_image(
                images,
                image_folder + "/train_{0}.jpg".format(step),
                nrow=16,
                normalize=True,
                value_range=(0, 1),
                scale_each=True,
            )

            config.update({"key": key, "iteration": iteration})

            save_model(
                params,
                opt_state,
                ckpt_manager,
                ema_params,
                config=config,
                step=step,
                unreplicate=True,
            )

    params, ema_params, opt_state = map(UNREPLICATE, (params, ema_params, opt_state))

    return params, ema_params, opt_state
