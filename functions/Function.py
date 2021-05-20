class Function:
    """Defines a function and its derivative.

    Attributes:
        f (function): An element-wise differentiable function that acts on a
            1-d numpy array of arbitrary dimension. May include a second
            argument for a label, e.g. for softmax-cross-entropy.
        f_prime (function): The element-wise derivative of f with respect to
            the first argument, must also act on 1-d numpy arrays of arbitrary
            dimension."""

    def __init__(self, f, f_prime):
        """Inits an instance of Function by specifying f and f_prime."""

        self.f = f
        self.f_prime = f_prime












