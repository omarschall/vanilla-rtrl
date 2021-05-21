from optimizers.Optimizer import Optimizer
import numpy as np

class SGD_Momentum(Optimizer):
    """Impelements SGD with classical momentum."""

    def __init__(self, lr=0.001, mu=0.8, **kwargs):

        allowed_kwargs_ = set()
        super().__init__(allowed_kwargs_, **kwargs)

        self.lr_ = np.copy(lr)
        self.lr = lr
        self.mu = mu
        self.vel = None

    def get_updated_params(self, params, grads):
        """Returns a list of updated parameter values (NOT the change in value).

        Args:
            params (list): List of trainable parameters as numpy arrays
            grads (list): List of corresponding gradients as numpy arrays.
        Returns:
            updated_params (list): List of newly updated parameters."""

        if self.lr_decay_rate is not None:
            self.lr = self.lr_decay()

        if self.clip_norm is not None:
            grads = self.clip_gradient(grads)

        if self.normalize:
            grads = self.normalize_gradient(grads)

        if self.vel is None:
            self.vel = [np.zeros_like(g) for g in params]

        self.vel = [self.mu * v - self.lr * g for v, g in zip(self.vel, grads)]

        updated_params = []
        for param, v in zip(params, self.vel):
            updated_params.append(param + v)

        return updated_params

