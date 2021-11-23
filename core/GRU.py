import numpy as np
from functions import *



class GRU:
    """A vanilla recurrent neural network.

    Obeys the forward equation (in TeX)

    a_t = (1 - \alpha)a_{t-1} + W_{rec}\phi(a_{t-1}) + W_{in}x_t + b_{rec}
    z_t = W_{out}a_t + b_out

    Attributes:
        n_in (int): Number of input dimensions
        n_h (int): Number of hidden units
        n_out (int): Number of output dimensions
        W_in (numpy array): Array of shape (n_h, n_in), weights task inputs.
        W_rec (numpy array): Array of shape (n_h, n_h), weights recurrent
            inputs.
        W_out (numpy array): Array of shape (n_out, n_h), provides hidden-to-
            output-layer weights.
        b_rec (numpy array): Array of shape (n_h), represents the bias term
            in the recurrent update equation.
        params (list): The list of each parameter's current value, in the order
            [W_rec, W_in, b_rec, W_out, b_out].
        shapes (list): The shape of each trainable set of parameters, in the
            same order.
        n_params (int): Number of total trainable parameters.
        n_h_params (int): Number of total trainble parameters in the recurrent
            update ewquation, i.e. all parameters excluding the output weights
            and biases.
        L2_indices (list): A list of integers representing which indices in
            the list of params should be subject to L2 regularization if the
            learning algorithm dictates (all weights but not biases).
        activation (functions.Function): An instance of the Function class
            used as the network's nonlinearity \phi in the recurrent update
            equation.
        alpha (float): Ratio of time constant of integration to time constant
            of leak. Must be less than 1.
        output (functions.Function): An instance of the Function class used
            for calculating final output from z.
        loss (functions.Function): An instance of the Function class used for
            calculating loss from z (must implicitly include output function,
            e.g. softmax_cross_entropy if output is softmax).
        x (numpy array): Array of shape (n_in) representing the current inputs
            to the network.
        h (numpy array): Array of shape (n_h) representing the pre-activations
            of the network.
        a (numpy array): Array of shape (n_h) representing the post-activations
            of the network.
        z (numpy array): Array of shape (n_out) reprenting the outputs of the
            network, before any final output nonlinearities, e.g. softmax,
            are applied.
        error (numpy array): Array of shape (n_out) representing the derivative
            of the loss with respect to z. Calculated by loss.f_prime.
        y_hat (numpy array): Array of shape (n_out) representing the final
            outputs of the network, to be directly compared with task labels.
            Not computed in any methods in this class.
        *_prev (numpy array): Array representing any of x, h, a, or z at the
            previous time step.
        a_J (numpy array): Array of shape (n_h, n_h) representing the Jacobian
            of the network at current time, based on the equation (in TeX)
            J_{ij} = \alpha\phi'(h_i) W_{rec,ij} + (1 - \alpha)\delta_{ij}."""
    def __init__(self, Wz_in, Wr_in, Wh_in, Wz_rec, Wr_rec, Wh_rec, W_out, bz_rec, br_rec, bh_rec, b_out,
                 activation, alpha, output, loss):

        # !!!: I add a new property for both GRU and RNN models so as to identify the model type
        self.type = "gru"
        self.Wz_in = Wz_in
        self.Wz_rec = Wz_rec
        self.Wr_in = Wr_in
        self.Wr_rec = Wr_rec
        self.Wh_in = Wh_in
        self.Wh_rec = Wh_rec
        self.W_out = W_out

        self.bz_rec = bz_rec
        self.br_rec = br_rec
        self.bh_rec = bh_rec
        self.b_out = b_out

        #Network dimensions
        self.n_in = Wz_in.shape[1]
        self.n_h = Wz_in.shape[0]
        self.n_out = W_out.shape[0]

        #Check dimension consistency.
        assert self.n_h == Wz_rec.shape[0]
        assert self.n_h == Wz_rec.shape[1]
        assert self.n_h == Wz_in.shape[0]
        assert self.n_h == W_out.shape[1]
        assert self.n_h == bz_rec.shape[0]
        assert self.n_out == b_out.shape[0]

        self.params = [self.Wz_rec, self.Wz_in, self.bz_rec,
                       self.Wr_rec, self.Wr_in, self.br_rec,
                       self.Wh_rec, self.Wh_in, self.bh_rec,
                       self.W_out, self.b_out]
        self.shapes = [w.shape for w in self.params]

        #Activation and loss functions
        self.alpha = alpha
        self.activation = activation
        self.output = output
        self.loss = loss

        #Number of parameters
        '''
        self.n_h_params = (self.W_rec.size +
                           self.W_in.size +
                           self.b_rec.size)
        self.n_params = (self.n_h_params +
                         self.W_out.size +
                         self.b_out.size)
        '''

        #Params for L2 regularization
        self.L2_indices = [0, 1, 3, 4, 6, 7, 9]

        #Initial state values
        self.reset_network()

    def reset_network(self, sigma=1, **kwargs):
        """Resets hidden state of the network, either randomly or by
        specifying with kwargs.

        Args:
            sigma (float): Standard deviation of (zero-mean) Gaussian random
                reset of pre-activations h. Used if neither h or a is
                specified.
            h (numpy array): The specification of the pre-activation values,
                must be of shape (self.n_h). The a values are determined by
                application of the nonlinearity to h.
            a (numpy array): The specification of the post-activation values,
                must be of shape (self.n_h). If not specified, determined
                by h."""

        if 'h' in kwargs.keys(): #Manual reset if specified.
            self.h = kwargs['h']
        else: #Random reset by sigma if not.
            self.h = np.random.normal(0, sigma, self.n_h)
            self.zz = np.random.normal(0, sigma, self.n_h)
            self.r = np.random.normal(0, sigma, self.n_h)

        self.a = self.activation.f(self.h) #Specify activations by \phi.

        if 'a' in kwargs.keys(): #Override with manual activations if given.
            self.a = kwargs['a']

        self.z = self.W_out.dot(self.a) + self.b_out #Specify outputs from a

    def z_out(self):
        """Update outputs using current state of the network."""

        self.z_prev = np.copy(self.z)
        self.z = self.W_out.dot(self.a) + self.b_out

    def next_state(self, x, a=None, update=True, sigma=0):
        """Advances the network forward by one time step.

        Accepts as argument the current time step's input x and updates
        the state of the RNN, while storing the previous state h
        and activatation a. Can either update the network (if update=True)
        or return what the update would be.

        Args:
            x (numpy array): Input provided to the network, of shape (n_in).
            update (bool): Specifies whether to update the network using the
                current network state (if True) or return the would-be next
                network state using a provided "current" network state a.
            a (numpy array): Recurrent inputs used to drive the network, to be
                provided only if update is False.
            sigma (float): Standard deviation of white noise added to pre-
                activations before applying \phi.

        Returns:
            Updates self.x, self.h, self.a, and self.*_prev, or returns the
            would-be update from given previous state a."""

        if update: #Update network if update is True
            self.x = x
            self.h_prev = np.copy(self.h)
            self.a_prev = np.copy(self.a)
            self.zz_prev = np.copy(self.zz)
            self.r_prev = np.copy(self.r)

            # !!!: I use zz instead of z because self.z is used for output
            # !!!: You can see the process of GRU here
            # !!!: In my understanding this sigmoid here is not flexible for change?
            # !!!: So I import functions and use it here
            self.zz = sigmoid.f(self.Wz_rec.dot(self.a) + self.Wz_in.dot(self.x) +
                      self.bz_rec)
            self.r = sigmoid.f(self.Wr_rec.dot(self.a) + self.Wr_in.dot(self.x) +
                      self.br_rec)
            self.h_ = self.activation.f(self.Wh_rec.dot(self.a*self.r) + self.Wh_in.dot(self.x) +
                      self.bh_rec)
            self.h = self.zz*self.a + (1-self.zz)*self.h_
            # !!!: Here I don't know if I still need self.alpha for recurrent update
            # !!!: when I have self.zz, so I keep them both for now.
            if sigma>0: #Add noise to h if sigma is more than 0.
                self.noise = sigma * np.random.normal(0, self.alpha, self.n_h)
                #self.h += self.noise
            else:
                self.noise = 0
            #Implement recurrent update equation
            self.a = ((1 - self.alpha)*self.a +
                      self.alpha*self.h) + self.noise
        else: #Otherwise calculate would-be next state from provided input a.
            z = sigmoid.f(self.Wz_rec.dot(a) + self.Wz_in.dot(x) + self.bz_rec)
            r = sigmoid.f(self.Wr_rec.dot(a) + self.Wr_in.dot(x) + self.br_rec)
            h_ = self.activation.f(self.Wh_rec.dot(a*r) + self.Wh_in.dot(x) + self.bh_rec) #Calculate new pre-activations
            h = z*a + (1-z)*h_
            ret = (1 - self.alpha)*a + self.alpha * h
            if sigma > 0:
                noise = np.random.normal(0, sigma, self.n_h)
                ret += noise
            return ret


    def get_a_jacobian(self, update=True, **kwargs):
        """Calculates the Jacobian of the network.

        Follows the equation
        J_{ij} = \alpha\phi'(h_i) W_{rec,ij} + (1 - \alpha)\delta_{ij}

        If update is True, the network attribute a_J updates to this Jacobian,
        based on current values of h and W_rec. If update is False, this method
        returns the Jacobian calculated based on current values of h and W_rec
        or specified values in kwargs if provided.

        Args:
            update (bool): Specifies whether to update or return the Jacobian.
            h (numpy array): Array of shape (n_h) that specifies what values of
                the pre-activations to use in calculating the Jacobian.
            W_rec (numpy array): Array of shape (n_h, n_h) that specifies
                what values of the recurrent weights to use in calculating
                the Jacobian."""

        #Use kwargs instead of defaults if provided
        if 'h' in kwargs.keys():
            h = kwargs['h']
            zz = kwargs['zz']
            r = kwargs['r']
        else:
            h = np.copy(self.h)
            zz = np.copy(self.zz)
            r = np.copy(self.r)
        if 'W_rec' in kwargs.keys():
            Wz_rec = kwargs['W_rec']
        else:
            Wz_rec = np.copy(self.Wz_rec)
            Wr_rec = np.copy(self.Wr_rec)
            Wh_rec = np.copy(self.Wh_rec)

        #Calculate Jacobian
        Dh = np.diag(self.activation.f_prime(h))
        Dzz = np.diag(self.activation.f_prime(h)*sigmoid.f_prime(zz))
        Dr =np.diag(self.activation.f_prime(h) * sigmoid.f_prime(zz) * sigmoid.f_prime(r))
        D = np.concatenate([Dzz,Dr,Dh])
        WW = np.concatenate([Wz_rec,Wr_rec,Wh_rec],axis=1)

        # !!!: Here I concatenate the derivative as a whole for later use
        a_J = self.alpha * D.dot(WW) + (1 - self.alpha) * np.eye(3*self.n_h)
        if update: #Update if update is True
            self.a_J = np.copy(a_J)
        else: #Otherwise return
            return a_J

    # !!!: I have very quick and intuitive implementations for functions below
    # !!!: and they are not tested yet.
    def get_network_speed(self, a=None):
        """Calculates and returns the (squared) 'speed' of the network given
        its current state and parameters. Option to specify a state value."""

        if a is None:
            a = self.a
        z = sigmoid.f(self.Wz_rec.dot(a) + self.bz_rec)
        r = sigmoid.f(self.Wr_rec.dot(a) + self.br_rec)
        h_ = self.activation.f(self.Wh_rec.dot(a * r) + self.bh_rec)
        h = z*a + (1-z)*h_
        delta_a = h - a

        return (self.alpha**2 / 2) * np.square(delta_a).sum()

    def get_network_speed_gradient(self, a=None):
        """Calculates and returns the gradient of the (squared) network speed
        with respect to the state of the network."""

        if a is None:
            a = self.a

        z = sigmoid.f(self.Wz_rec.dot(a) + self.bz_rec)
        r = sigmoid.f(self.Wr_rec.dot(a) + self.br_rec)
        h_ = self.Wh_rec.dot(a * r) + self.bh_rec
        phi = z*a + (1-z)*self.activation.f(h_)
        D = self.activation.f_prime(h_)
        delta_a = phi - a
        delta_w = (D * self.Wh_rec.T).T - np.eye(self.n_h)
        ret = delta_a.dot(delta_w)

        return (self.alpha**2) * ret

    def get_network_speed_gradient_wrt_weights(self, a=None):
        """Calculates and returns the gradient of the (squared) network speed
        with respect to the network parameters."""

        if a is None:
            a = self.a

        z = sigmoid.f(self.Wz_rec.dot(a) + self.bz_rec)
        r = sigmoid.f(self.Wr_rec.dot(a) + self.br_rec)
        h_ = self.Wh_rec.dot(a * r) + self.bh_rec
        phi = z*a + (1-z)*self.activation.f(h_)
        D = self.activation.f_prime(h_)
        delta_a = phi - a

        x = np.zeros_like(self.x)
        a_hat = np.concatenate([a, x, np.array([1])])

        return (self.alpha**2) * (np.outer(D, a_hat).T * delta_a).T
