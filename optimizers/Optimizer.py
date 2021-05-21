import numpy as np

class Optimizer:
    """Parent class for gradient-based optimizers."""

    def __init__(self, allowed_kwargs_, **kwargs):

        allowed_kwargs = {'lr_decay_rate', 'min_lr',
                          'clip_norm', 'normalize',
                          'rec_proj_mats', 'out_proj_mats'}.union(allowed_kwargs_)
        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError('Unexpected keyword argument '
                                'passed to Optimizer: ' + str(k))

        #Set all non-specified kwargs to None
        for attr in allowed_kwargs:
            if not hasattr(self, attr):
                setattr(self, attr, None)

        self.__dict__.update(kwargs)

    def clip_gradient(self, grads):
        """Clips each gradient by the global gradient norm if it exceeds
        self.clip_norm.

        Args:
            grads (list): List of original gradients
        Returns:
            clipped_grads (list): List of clipped gradients."""

        grad_norm = np.sqrt(sum([np.square(grad).sum() for grad in grads]))
        if grad_norm > self.clip_norm:
            clipped_grads = []
            for grad in grads:
                clipped_grads.append(grad * (self.clip_norm/grad_norm))
            return clipped_grads
        else:
            return grads

    def normalize_gradient(self, grads):
        """Takes in a list of gradients and forces the overall norm to be
        unity."""

        grad_norm = np.sqrt(sum([np.square(grad).sum() for grad in grads]))
        normalized_grads = []
        for grad in grads:
            normalized_grads.append(grad / grad_norm)
        return normalized_grads

    def lr_decay(self):
        """Multiplicatively decays the learning rate by a factor of
        self.lr_decay_rate, with a floor learning rate of self.min_lr."""

        self.lr_ = self.lr_ * self.lr_decay_rate
        try:
            return np.max([self.lr_, self.min_lr])
        except AttributeError:
            return self.lr_