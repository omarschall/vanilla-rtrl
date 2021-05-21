from optimizers.Optimizer import Optimizer
import numpy as np

class Adam(Optimizer):

    """Adam optimizer.
    Default parameters follow those provided in the original paper.
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor.
        decay: float >= 0. Learning rate decay over each update.
    # References
        - [Adam - A Method for Stochastic Optimization]
            (http://arxiv.org/abs/1412.6980v8)"""

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, decay=0., **kwargs):

        allowed_kwargs_ = set()
        super().__init__(allowed_kwargs_, **kwargs)

        self.iterations = 0
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.decay = decay
        self.epsilon = epsilon
        self.initial_decay = decay

    def get_updated_params(self, params, grads):
        """ params and grads are list of numpy arrays
        """
        original_shapes = [x.shape for x in params]
        params = [x.flatten() for x in params]
        grads = [x.flatten() for x in grads]

        if hasattr(self, 'clipnorm') and self.clipnorm > 0:
            norm = np.sqrt(sum([np.sum(np.square(g)) for g in grads]))
            grads = [self.clip_norm(g, norm) for g in grads]

        '''
        if hasattr(self, 'clipvalue') and self.clipvalue > 0:
            grads = [K.clip(g, -self.clipvalue, self.clipvalue) for g in grads]
        '''

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))

        t = self.iterations + 1
        lr_t = lr * (np.sqrt(1. - np.power(self.beta_2, t)) /
                     (1. - np.power(self.beta_1, t)))

        if not hasattr(self, 'ms'):
            self.ms = [np.zeros(p.shape) for p in params]
            self.vs = [np.zeros(p.shape) for p in params]

        ret = [None] * len(params)
        self.vel = [None] * len(params)
        for i, p, g, m, v in zip(range(len(params)), params, grads, self.ms, self.vs):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * np.square(g)
            self.vel[i] = - lr_t * m_t / (np.sqrt(v_t) + self.epsilon)
            p_t = p + self.vel[i]
            self.ms[i] = m_t
            self.vs[i] = v_t
            ret[i] = p_t

        self.iterations += 1

        for i in range(len(ret)):
            ret[i] = ret[i].reshape(original_shapes[i])

        return ret

