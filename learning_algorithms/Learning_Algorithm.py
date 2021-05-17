
from utils import *
from functions import *


class Learning_Algorithm:
    """Parent class for all learning algorithms.

    Attributes:
        rnn (network.RNN): An instance of RNN to be trained by the network.
        n_* (int): Extra pointer to rnn.n_* (in, h, out) for conveneince.
        m (int): Number of recurrent "input dimensions" n_h + n_in + 1 including
            task inputs and constant 1 for bias.
        q (numpy array): Array of immediate error signals for the hidden units,
            i.e. the derivative of the current loss with respect to rnn.a, of
            shape (n_h).
        W_FB (numpy array or None): A fixed set of weights that may be provided
            for an approximate calculation of q in the manner of feedback
            alignment (Lillicrap et al. 2016).
        L2_reg (float or None): Strength of L2 regularization parameter on the
            network weights.
        a_ (numpy array): Array of shape (n_h + 1) that is the concatenation of
            the network's state and the constant 1, used to calculate the output
            errors.
        q (numpy array): The immediate loss derivative of the network state
            dL/da, calculated by propagate_feedback_to_hidden.
        q_prev (numpy array): The q value from the previous time step."""

    def __init__(self, rnn, allowed_kwargs_=set(), **kwargs):
        """Initializes an instance of learning algorithm by specifying the
        network to be trained, custom allowable kwargs, and kwargs.

        Args:
            rnn (network.RNN): An instance of RNN to be trained by the network.
            allowed_kwargs_ (set): Set of allowed kwargs in addition to those
                common to all child classes of Learning_Algorithm, 'W_FB' and
                'L2_reg'."""

        allowed_kwargs = {'W_FB', 'L1_reg', 'L2_reg', 'CL_method',
                          'maintain_sparsity'}.union(allowed_kwargs_)

        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError('Unexpected keyword argument passed'
                                'to Learning_Algorithm.__init__: ' + str(k))

        # Set all non-specified kwargs to None
        for attr in allowed_kwargs:
            if not hasattr(self, attr):
                setattr(self, attr, None)

        # Make kwargs attributes of the instance
        self.__dict__.update(kwargs)

        # Define basic learning algorithm properties
        self.rnn = rnn
        self.n_in = self.rnn.n_in
        self.n_h = self.rnn.n_h
        self.n_out = self.rnn.n_out
        self.m = self.n_h + self.n_in + 1
        self.q = np.zeros(self.n_h)

    def get_outer_grads(self):
        """Calculates the derivative of the loss with respect to the output
        parameters rnn.W_out and rnn.b_out.

        Calculates the outer gradients in the manner of a perceptron derivative
        by taking the outer product of the error with the "regressors" onto the
        output (the hidden state and constant 1).

        Returns:
            A numpy array of shape (rnn.n_out, self.n_h + 1) containing the
                concatenation (along column axis) of the derivative of the loss
                w.r.t. rnn.W_out and w.r.t. rnn.b_out."""

        self.a_ = np.concatenate([self.rnn.a, np.array([1])])
        return np.multiply.outer(self.rnn.error, self.a_)

    def propagate_feedback_to_hidden(self):
        """Performs one step of backpropagation from the outer-layer errors to
        the hidden state.

        Calculates the immediate derivative of the loss with respect to the
        hidden state rnn.a. By default, this is done by taking rnn.error (dL/dz)
        and applying the chain rule, i.e. taking its matrix product with the
        derivative dz/da, which is rnn.W_out. Alternatively, if 'W_FB' attr is
        provided to the instance, then these feedback weights, rather the W_out,
        are used, as in feedback alignment. (See Lillicrap et al. 2016.)

        Updates q to the current value of dL/da."""

        self.q_prev = np.copy(self.q)

        if self.W_FB is None:
            self.q = self.rnn.error.dot(self.rnn.W_out)
        else:
            self.q = self.rnn.error.dot(self.W_FB)

    def L2_regularization(self, grads):
        """Adds L2 regularization to the gradient.

        Args:
            grads (list): List of numpy arrays representing gradients before L2
                regularization is applied.
        Returns:
            A new list of grads with L2 regularization applied."""

        # Get parameters affected by L2 regularization
        L2_params = [self.rnn.params[i] for i in self.rnn.L2_indices]
        # Add to each grad the corresponding weight's current value, weighted
        # by the L2_reg hyperparameter.
        for i_L2, W in zip(self.rnn.L2_indices, L2_params):
            grads[i_L2] += self.L2_reg * W
        # Calculate L2 loss for monitoring purposes
        self.L2_loss = 0.5 * sum([norm(p) ** 2 for p in L2_params])
        return grads

    def L1_regularization(self, grads):
        """Adds L1 regularization to the gradient.

        Args:
            grads (list): List of numpy arrays representing gradients before L1
                regularization is applied.
        Returns:
            A new list of grads with L1 regularization applied."""

        # Get parameters affected by L1 regularization
        # (identical as those affected by L2)
        L1_params = [self.rnn.params[i] for i in self.rnn.L2_indices]
        # Add to each grad the sign of the corresponding parameter weighted
        # by L1 reg strength
        for i_L1, W in zip(self.rnn.L2_indices, L1_params):
            grads[i_L1] += self.L1_reg * np.sign(W)
        # Calculate L2 loss for monitoring purposes
        self.L1_loss = sum([norm(p) for p in L1_params])
        return grads

    def apply_sparsity_to_grads(self, grads):
        """"If called, modifies gradient to make 0 any parameters that are
        already 0 (only for L2 params).

        Args:
            grads (list): List of numpy arrays representing gradients before L2
                regularization is applied.
        Returns:
            A new list of grads with L2 regularization applied."""

        # Get parameters affected by L2 regularization
        L2_params = [self.rnn.params[i] for i in self.rnn.L2_indices]
        # AMultiply each gradient by 0 if it the corresponding weight is
        # already 0
        for i_L2, W in zip(self.rnn.L2_indices, L2_params):
            grads[i_L2] *= (W != 0)
        return grads

    def __call__(self):
        """Calculates the final list of grads for this time step.

        Assumes the user has already called self.update_learning_vars, a
        method specific to each child class of Real_Time_Learning_Algorithm
        that updates internal learning variables, e.g. the influence matrix of
        RTRL. Then calculates the outer grads (gradients of W_out and b_out),
        updates q using propagate_feedback_to_hidden, and finally calling the
        get_rec_grads method (specific to each child class) to get the gradients
        of W_rec, W_in, and b_rec as one numpy array with shape (n_h, m). Then
        these gradients are split along the column axis into a list of 5
        gradients for W_rec, W_in, b_rec, W_out, b_out. L2 regularization is
        applied if L2_reg parameter is not None.

        Returns:
            List of gradients for W_rec, W_in, b_rec, W_out, b_out."""

        self.outer_grads = self.get_outer_grads()
        self.propagate_feedback_to_hidden()
        self.rec_grads = self.get_rec_grads()
        rec_grads_list = split_weight_matrix(self.rec_grads,
                                             [self.n_h, self.n_in, 1])
        outer_grads_list = split_weight_matrix(self.outer_grads,
                                               [self.n_h, 1])
        grads_list = rec_grads_list + outer_grads_list

        if self.L1_reg is not None:
            grads_list = self.L1_regularization(grads_list)

        if self.L2_reg is not None:
            grads_list = self.L2_regularization(grads_list)

        if self.CL_method is not None:
            grads_list = self.CL_method(grads_list)

        if self.maintain_sparsity:
            grads_list = self.apply_sparsity_to_grads(grads_list)

        return grads_list

    def reset_learning(self):
        """Resets internal variables of the learning algorithm (relevant if
        simulation includes a trial structure). Default is to do nothing."""

        pass
