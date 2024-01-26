import os
import jax
import flax
import optax
from einops import rearrange
from flax.core.frozen_dict import FrozenDict

import orbax.checkpoint as checkpoint
from flax.training import orbax_utils

NUM_DEVICE = len(jax.local_devices())


def unshard(x: jax.Array) -> jax.Array:
    return rearrange(x, "d b ... -> (d b) ...")


def shard(x: jax.Array) -> jax.Array:
    valid_size = x.shape[0] - (x.shape[0] - (x.shape[0] // NUM_DEVICE * NUM_DEVICE))
    return rearrange(x[:valid_size], "(d b) ... -> d b ...", d=NUM_DEVICE)


def get_nparams(params: FrozenDict) -> int:
    nparams = 0
    for item in params:
        if isinstance(params[item], FrozenDict) or isinstance(params[item], dict):
            nparams += get_nparams(params[item])
        else:
            nparams += params[item].size
    return nparams


def make_folder(name: str):
    if not (os.path.exists(name)):
        os.makedirs(name)


def restore_model(
    checkpoint_manager: checkpoint.CheckpointManager,
    target: dict = None,
    latest_step: int = None,
):
    if latest_step is None:
        latest_step = checkpoint_manager.latest_step()

    assert latest_step > 0

    if target is None:
        return checkpoint_manager.restore(latest_step)
    return checkpoint_manager.restore(latest_step, items=target)


def create_checkpoint_manager(store_path: str = "./ckpts/stored", max_to_keep: int = 5):
    orbax_checkpointer = checkpoint.PyTreeCheckpointer()

    options = checkpoint.CheckpointManagerOptions(max_to_keep=max_to_keep, create=True)

    checkpoint_manager = checkpoint.CheckpointManager(
        store_path, orbax_checkpointer, options
    )
    return checkpoint_manager


def save_model(
    params: FrozenDict,
    opt_state: FrozenDict,
    checkpointer: checkpoint.CheckpointManager,
    ema_params: FrozenDict = None,
    unreplicate: bool = False,
    step: int = 0,
    config: dict = None,
):
    def fn(var):
        if var is not None:
            return flax.jax_utils.unreplicate(var)
        return var

    if unreplicate:
        params, opt_state, ema_params = map(fn, (params, opt_state, ema_params))

    ckpt = {
        "params": params,
        "ema_params": ema_params,
        "opt_state": opt_state,
        "config": config,
    }

    save_args = orbax_utils.save_args_from_target(ckpt)
    checkpointer.save(step, ckpt, save_kwargs={"save_args": save_args})
