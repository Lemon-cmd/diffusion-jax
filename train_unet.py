import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    return False


def str2tuple(v):
    return tuple([int(c) for c in v.split(",")])


parser = argparse.ArgumentParser(
    description="UNET Training Script for MNIST and CIFAR10/CIFAR100",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--clear_gpu_cache",
    default=False,
    type=str2bool,
    help="Flag to determine whether to clear gpu memory. Default is False assuming that you are not using a shared system.",
)

parser.add_argument(
    "--loss_type",
    default="pred_v",
    type=str,
    help='Available loss types: \{"pred_v", "pred_x0", "pred_noise" \}',
)

parser.add_argument(
    "--min_snr_gamma",
    default=5.0,
    type=float,
    help="Minimum signal to noise ratio for scaling loss weight.",
)

parser.add_argument(
    "--timesteps",
    default=1000,
    type=int,
    help="Number of timesteps used for training the model.",
)

parser.add_argument(
    "--sampling_steps",
    default=250,
    type=int,
    help="Number of timesteps used for sampling.",
)

parser.add_argument(
    "--seed",
    default=3867,
    type=int,
    help="Initial seed for randomization.",
)

parser.add_argument(
    "--save_every_k",
    default=5,
    type=int,
    help="The interval rate of which the model's parameters are saved and evaluation is performed.",
)

parser.add_argument(
    "--max_to_keep",
    default=5,
    type=int,
    help="The maximum number of model's parameters history to keep. Default is 5 copies.",
)

parser.add_argument(
    "--epochs",
    default=1000,
    type=int,
    help="The number of training points.",
)

parser.add_argument(
    "--batch_size",
    default=128,
    type=int,
    help="Batch size.",
)

parser.add_argument(
    "--num_workers",
    default=0,
    type=int,
    help="Number of workers for data loader.",
)

parser.add_argument(
    "--gradient_accummulation_steps",
    default=1,
    type=int,
    help="Number of gradient accummulation steps. Default is 1. If you're working under a limited system, try scaling down batch size and increasing gradient accummulation steps.",
)

parser.add_argument(
    "--eta",
    default=0.0,
    type=float,
    help="Stochasticity control variable for DDIM sampling.",
)

parser.add_argument(
    "--learning_rate",
    default=1e-4,
    type=float,
    help="Learning rate.",
)

parser.add_argument(
    "--weight_decay",
    default=1e-4,
    type=float,
    help="Weight decay value.",
)

parser.add_argument(
    "--max_ema_decay",
    default=0.9999,
    type=float,
    help="Maximum value for EMA decay.",
)

parser.add_argument(
    "--min_ema_decay",
    default=0.0,
    type=float,
    help="Minimum value for EMA decay.",
)

parser.add_argument(
    "--ema_decay_power",
    default=2 / 3,
    type=float,
    help="Decay power for EMA annealing.",
)


parser.add_argument(
    "--ema_inv_gamma",
    default=1.0,
    type=float,
    help="Inv gamma for EMA annealing",
)


parser.add_argument(
    "--start_ema_update_after",
    default=100,
    type=int,
    help="The number of parameters updates have to be performed before starting EMA update.",
)

parser.add_argument(
    "--update_ema_every",
    default=10,
    type=int,
    help="The interval in which EMA update occurs once it is initiated.",
)

parser.add_argument(
    "--result_path",
    default="./unet",
    type=str,
    help="Folder path to save model and results.",
)

parser.add_argument(
    "--root_folder",
    default="../data",
    type=str,
    help="Folder path to data.",
)

parser.add_argument(
    "--dataset",
    default="CIFAR10",
    type=str,
    help="Dataset name: \{CIFAR10, CIFAR100, MNIST \}",
)

parser.add_argument(
    "--beta_schedule",
    default="cosine",
    type=str,
    help='Variance scheduler: \{"linear", "cosine", "sigmoid"\}',
)

parser.add_argument(
    "--dim",
    default=64,
    type=int,
    help="Embedding dim for Unet.",
)

parser.add_argument(
    "--dim_mults",
    default=(1, 2, 4, 8),
    type=str2tuple,
    help="Dim. multipliers for Unet.",
)

parser.add_argument(
    "--resnet_block_groups",
    default=8,
    type=int,
    help="Resnet block groups number.",
)

parser.add_argument(
    "--out_dim",
    default=None,
    type=int,
    help="Output dim. of Unet. Default is None for reconstruction of input.",
)

parser.add_argument(
    "--init_dim",
    default=None,
    type=int,
    help="Initial encoding dimension for latent variable. Default is None to set equal to dim. number.",
)

parser.add_argument(
    "--learned_variance",
    default=False,
    type=str2bool,
    help="Learn variance according to ddpm paper.",
)

args = parser.parse_args()
config = vars(args)
import pprint

pprint.pprint(config, width=1)


import os

if config["clear_gpu_cache"]:
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


import jax
import copy
import optax
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10, CIFAR100

from src import (
    linear_schedule,
    cosine_schedule,
    sigmoid_schedule,
    get_var_params,
    DiffusionLoader,
    train_model,
    get_nparams,
    Unet,
)

if config["dataset"].lower() == "mnist":
    data = MNIST
elif config["dataset"].lower() == "cifar10":
    data = CIFAR10
elif config["dataset"].lower() == "cifar100":
    data = CIFAR100
else:
    raise Exception(
        "For this script, we only utilize MNIST and CIFAR datasets. If you wish to try something else, please edit the script."
    )

if config["beta_schedule"].lower() == "linear":
    beta_schedule = linear_schedule
elif config["beta_schedule"].lower() == "cosine":
    beta_schedule = cosine_schedule
elif config["beta_schedule"].lower() == "sigmoid":
    beta_schedule = sigmoid_schedule
else:
    raise Exception("linear, cosine, and sigmoid are the only available options.")


betas = beta_schedule(config["timesteps"])
var_params = get_var_params(betas)


train_set = data(
    root="../data",
    download=True,
    train=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    ),
)

train_loader = DataLoader(
    train_set,
    shuffle=True,
    batch_size=config["batch_size"],
    num_workers=config["num_workers"],
)

train_loader = DiffusionLoader(
    train_loader,
    var_params,
    loss_type=config["loss_type"],
    min_snr_gamma=config["min_snr_gamma"],
)

model = Unet(
    dim=config["dim"],
    init_dim=config["init_dim"],
    out_dim=config["out_dim"],
    dim_mults=config["dim_mults"],
    resnet_block_groups=config["resnet_block_groups"],
    learned_variance=config["learned_variance"],
)

x, _, t, _ = train_loader()
key = jax.random.PRNGKey(config["seed"])

config["key"] = key
config["iteration"] = 0
config["var_params"] = var_params
config["result_path"] = config["result_path"] + "_" + config["loss_type"]

params = model.init(key, x[:1], t[:1])["params"]

ema_params = copy.deepcopy(params)
print("Number of params: ", f"{get_nparams(params):,}")

opt = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(
        config["learning_rate"],
        b1=0.9,
        b2=0.99,
        weight_decay=config["weight_decay"],
    ),
)
opt = optax.MultiSteps(opt, every_k_schedule=config["gradient_accummulation_steps"])

opt_state = opt.init(params)

params, ema_params, opt_state = train_model(
    model, params, ema_params, opt, opt_state, train_loader, config=config
)
