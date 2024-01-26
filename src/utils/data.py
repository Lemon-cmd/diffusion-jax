import numpy as np
from typing import Tuple, Optional
from torch.utils.data import DataLoader

from .sampling import predict_v_from_x0


def normalize_to_neg_one_to_one(t):
    # assuming t is between 0 and 1
    return 2 * t - 1


class DiffusionLoader(object):
    def __init__(
        self,
        loader: DataLoader,
        var_params: dict,
        *,
        min_snr_gamma: Optional[float] = None,
        loss_type: str = "pred_v"
    ) -> None:
        super().__init__()

        self.size = len(loader)
        self.loader = loader
        self.var_params = var_params
        self.alphas_cp = var_params["alphas_cp"]
        self.snr = self.alphas_cp / (1.0 - self.alphas_cp)
        self.loss_weight = np.copy(self.snr)

        if min_snr_gamma is not None:
            self.loss_weight = np.clip(
                self.loss_weight,
                None,
                min_snr_gamma,
            )

        self.loss_type = loss_type
        if loss_type == "pred_v":
            self.loss_weight = self.loss_weight / (self.snr + 1)
        elif loss_type == "pred_noise":
            self.loss_weight = self.loss_weight / (self.snr)
        elif loss_type == "pred_x0":
            pass
        else:
            raise Exception("Invalid Loss Type")

    def __len__(self):
        return self.size

    def __get_data(self):
        loader_iter = iter(self.loader)
        try:
            x, _ = next(loader_iter)

        except StopIteration:
            loader_iter = iter(self.loader)
            x, _ = next(loader_iter)

        x = normalize_to_neg_one_to_one(x)
        return x.permute(0, 2, 3, 1).numpy()  # B, H, W, C

    def __call__(self, timesteps: int = 1000) -> Tuple[np.ndarray]:
        assert timesteps > 0
        timesteps = min(timesteps, self.alphas_cp.shape[0])

        x0 = self.__get_data()
        t = np.random.randint(0, timesteps, size=x0.shape[0])

        alphas_t_cp = self.alphas_cp[t]
        loss_weight = self.loss_weight[t]
        eps = np.random.normal(0, 1, size=x0.shape)
        x_t = np.sqrt(alphas_t_cp) * x0 + np.sqrt(1.0 - alphas_t_cp) * eps

        if self.loss_type == "pred_noise":
            labels = eps
        elif self.loss_type == "pred_x0":
            labels = x0
        else:
            labels = predict_v_from_x0(x0, eps, t, self.var_params)

        assert x0.shape == x_t.shape

        # return: inputs, labels, times, loss_weight
        return x_t, labels, t, loss_weight
