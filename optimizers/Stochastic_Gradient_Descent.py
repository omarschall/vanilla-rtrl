from optimizers.Optimizer import Optimizer
import numpy as np

class Stochastic_Gradient_Descent(Optimizer):
    """Implements basic stochastic gradient descent optimizer.

    Attributes:
        lr (float): learning rate."""

    def __init__(self, lr=0.001, **kwargs):

        allowed_kwargs_ = set()
        super().__init__(allowed_kwargs_, **kwargs)

        self.lr_ = np.copy(lr)
        self.lr = lr

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

        updated_params = []
        self.vel = [-self.lr * g for g in grads]
        for param, v in zip(params, self.vel):
            updated_params.append(param + v)

        return updated_params

