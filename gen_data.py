#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 16:30:03 2018

@author: omarschall
"""

import numpy as np
import pickle

class Task:
    """Parent class for all tasks. A Task is a class whose instances generate
    datasets to be used for training RNNs.

    A dataset is a dict of dicts with
    keys 'train' and 'test', which point to dicts with keys 'X' and 'Y' for
    inputs and labels, respectively. The values for 'X' and 'Y' are numpy
    arrays with shapes (time_steps, n_in) and (time_steps, n_out),
    respectively."""

    def __init__(self, n_in, n_out):
        """Initializes a Task with the number of input and output dimensions

        Args:
            n_in (int): Number of input dimensions.
            n_out (int): Number of output dimensions."""

        self.n_in = n_in
        self.n_out = n_out

    def gen_data(self, N_train, N_test):
        """Generates a data dict with a given number of train and test examples.

        Args:
            N_train (int): number of training examples
            N_test (int): number of testing examples
        Returns:
            data (dict): Dictionary pointing to 2 sub-dictionaries 'train'
                and 'test', each of which has keys 'X' and 'Y' for inputs
                and labels, respectively."""

        data = {'train': {}, 'test': {}}

        data['train']['X'], data['train']['Y'] = self.gen_dataset(N_train)
        data['test']['X'], data['test']['Y'] = self.gen_dataset(N_test)

        return data

    def gen_dataset(self, N):
        """Function specific to each class, randomly generates a dictionary of
        inputs and labels.

        Args:
            N (int): number of examples
        Returns:
            dataset (dict): Dictionary with keys 'X' and 'Y' pointing to inputs
                and labels, respectively."""

        pass

class Add_Task(Task):
    """Class for the 'Add Task', an input-label mapping with i.i.d. Bernoulli
    inputs (p=0.5) and labels depending additively on previous inputs at
    t_1 and t_2 time steps ago:

    y(t) = 0.5 + 0.5 * x(t - t_1) - 0.25 * x(t - t_2)           (1)

    as inspired by Pitis 2016
    (https://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html).

    The inputs and outputs each have a redundant dimension representing the
    complement of the outcome (i.e. x_1 = 1 - x_0), because keeping all
    dimensions above 1 makes python broadcasting rules easier."""

    def __init__(self, t_1, t_2, deterministic=False, tau_task=1):
        """Initializes an instance of this task by specifying the temporal
        distance of the dependencies, whether to use deterministic labels, and
        the timescale of the changes.

        Args:
            t_1 (int): Number of time steps for first dependency
            t_2 (int): Number of time steps for second dependency
            deterministic (bool): Indicates whether to take the labels as
                the exact numbers in Eq. (1) OR to use those numbers as
                probabilities in Bernoulli outcomes.
            tau_task (int): Factor by which we temporally 'stretch' the task.
                For example, if tau_task = 3, each input (and label) is repeated
                for 3 time steps before being replaced by a new random
                sample."""

        #Initialize a parent Task object with 2 input and 2 output dimensions.
        super().__init__(2, 2)

        #Dependencies in coin task
        self.t_1 = t_1
        self.t_2 = t_2
        self.tau_task = tau_task

        #Use coin flip outputs or deterministic probabilities as labels
        self.deterministic = deterministic

    def gen_dataset(self, N):
        """Generates a dataset according to Eq. (1)."""

        #Generate random bernoulli inputs and labels according to Eq. (1).
        N = N // self.tau_task
        x = np.random.binomial(1, 0.5, N)
        y = 0.5 + 0.5 * np.roll(x, self.t_1) - 0.25 * np.roll(x, self.t_2)
        if not self.deterministic:
            y = np.random.binomial(1, y, N)
        X = np.array([x, 1 - x]).T
        Y = np.array([y, 1 - y]).T

        #Temporally stretch according to the desire timescale of change.
        X = np.tile(X, self.tau_task).reshape((self.tau_task*N, 2))
        Y = np.tile(Y, self.tau_task).reshape((self.tau_task*N, 2))

        return X, Y

class Mimic_RNN(Task):
    """Class for the 'Mimic Task,' where the inputs are random i.i.d. Bernoulli
    and the labels are the outputs of a fixed 'target' RNN that is fed these
    inputs."""

    def __init__(self, rnn, p_input, tau_task=1, latent_dim=None):
        """Initializes the task with a target RNN (instance of network.RNN),
        the probability of the Bernoulli inputs, and a time constant of change.

        Args:
            rnn (network.RNN instance): The target RNN
            p_input (float): The probability of any input having value 1
            tau_task (int): The temporal stretching factor for the inputs, see
                tau_task in Add_Task."""

        #Initialize as Task object with dims inherited from the target RNN.
        super().__init__(rnn.n_in, rnn.n_out)

        self.rnn = rnn
        self.p_input = p_input
        self.tau_task = tau_task
        self.latent_dim = latent_dim
        if self.latent_dim is not None:
            self.segment_length = self.n_in // self.latent_dim

    def gen_dataset(self, N):
        """Generates a dataset by first generating inputs randomly by the
        binomial distribution and temporally stretching them by tau_task,
        then feeding these inputs to the target RNN."""

        #Generate inputs
        N = N // self.tau_task
        X = []
        for i in range(N):
            if self.latent_dim is not None:
                outcomes = np.random.binomial(1, self.p_input, self.latent_dim)
                x = [o * np.ones(self.segment_length) for o in outcomes]
                x = np.concatenate(x)
            else:
                x = np.random.binomial(1, self.p_input, self.n_in)
            X.append(x)
        X = np.tile(X, self.tau_task).reshape((self.tau_task*N, self.n_in))

        #Get RNN responses
        Y = []
        self.rnn.reset_network(h=np.zeros(self.rnn.n_h))
        for i in range(len(X)):
            self.rnn.next_state(X[i])
            self.rnn.z_out()
            Y.append(self.rnn.output.f(self.rnn.z))

        return X, np.array(Y)

class Sequential_MNIST(Task):
    """Class for the Sequential MNIST task, where chunks of a given MNIST
    image are fed into the network at each time step. The number of pixels
    per time step parameterizes the difficulty of the task.

    Note: uses the numpy mnist pickle file mnist.pkl as created from the
    GitHub repo https://github.com/hsjeong5/MNIST-for-Numpy/."""

    def __init__(self, pixels_per_time_step):
        """Inits an instance of Sequential_MNIST by specifying the number
        of pixels per time step."""

        if (784 % pixels_per_time_step) != 0:
            raise ValueError('The number of pixels per time step must ' +
                             'divide the total pixels per image')

        super().__init__(pixels_per_time_step, 10)

        self.pixels_per_time_step = pixels_per_time_step
        self.inputs_per_image = 784 // self.pixels_per_time_step

        try:
            with open('library/mnist.pkl', 'rb') as f:
                mnist = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError('Must download mnist pickle file and' +
                                    'store in ./library/, see class docs')

        self.mnist_images = np.concatenate([mnist['training_images'],
                                            mnist['test_images']], axis=0)
        self.mnist_labels = np.concatenate([mnist['training_labels'],
                                            mnist['test_labels']])

    def gen_dataset(self, N):
        """Generates a test set by taking the concatenated training/test
        images and labels, randomly sampling from each, and then reshaping
        them into the form specified by pixels_per_time_step."""

        N_images = N // self.inputs_per_image
        image_indices = np.random.choice(list(range(70000)), N_images,
                                         replace=False)
        mnist_images = self.mnist_images[image_indices]
        mnist_labels = self.mnist_labels[image_indices]

        mnist_images = mnist_images.reshape((-1, self.pixels_per_time_step))
        one_hot_labels = np.squeeze(np.eye(10)[mnist_labels])

        X = mnist_images
        Y = np.tile(one_hot_labels, self.inputs_per_image).reshape((-1, 10))

        return X, Y

class Sensorimotor_Mapping(Task):
    """Class for a simple working memory task, where a binary input is given
    to the network, which it must report back some number of time steps later.

    In more detail, on a given trial, the input channel are 0 except during the
    stimulus (a user-defined time window), when it is either 1 or -1; or during
    the report period (another user-defined time window), when it is 1. The
    label channel is always 0 except during the

    Unlike other tasks, this task has a *trial* structure where the user has
    the option of resetting the network and/or learning algorithm states between
    trials."""

    def __init__(self, t_stim=1, stim_duration=3,
                 t_report=20, report_duration=3,
                 off_report_loss_factor=0.1):
        """Initializes an instance by specifying, within a trial, the time step
        where the stimulus occurs, the stimulus duration, the time of the
        report, the report duration, and the factor by which the loss is scaled
        for on- vs. off-report time steps.

        Args:
            t_stim (int): The time step when the stimulus starts
            stim_duration (int): The duration of the stimulus in time steps
            t_report (int): The time step when the report period starts
            report_duration (int): The report duration in time steps.
            off_report_loss_factor (float): A number between 0 and 1 that scales
                the loss function on time steps outside the report period."""

        super().__init__(2, 2)

        self.t_stim = t_stim
        self.stim_duration = stim_duration
        self.t_report = t_report
        self.report_duration = report_duration
        self.time_steps_per_trial = t_report + report_duration
        self.off_report_loss_factor = off_report_loss_factor

        #Make mask for preferential learning within task
        self.trial_mask = np.ones(self.time_steps_per_trial)
        self.trial_mask *= self.off_report_loss_factor
        self.trial_mask[self.t_report:self.t_report + self.report_duration] = 1

    def gen_dataset(self, N):
        """Generates a dataset by hardcoding in the two trial types, randomly
        generating a sequence of choices, and concatenating them at the end."""

        X = []
        Y = []

        #Trial type 1
        x_1 = np.zeros((self.time_steps_per_trial, 2))
        y_1 = np.ones_like(x_1)*0.5
        x_1[self.t_stim:self.t_stim + self.stim_duration, 0] = 1
        x_1[self.t_report:self.t_report + self.report_duration, 1] = 1
        y_1[self.t_report:self.t_report + self.report_duration, 0] = 1
        y_1[self.t_report:self.t_report + self.report_duration, 1] = 0

        #Trial type 2
        x_2 = np.zeros((self.time_steps_per_trial, 2))
        y_2 = np.ones_like(x_2)*0.5
        x_2[self.t_stim:self.t_stim + self.stim_duration, 0] = -1
        x_2[self.t_report:self.t_report + self.report_duration, 1] = 1
        y_2[self.t_report:self.t_report + self.report_duration, 0] = 0
        y_2[self.t_report:self.t_report + self.report_duration, 1] = 1

        x_trials = [x_1, x_2]
        y_trials = [y_1, y_2]

        N_trials = N // self.time_steps_per_trial
        for i in range(N_trials):

            trial_type = np.random.choice([0, 1])
            X.append(x_trials[trial_type])
            Y.append(y_trials[trial_type])

        if N_trials > 0:
            X = np.concatenate(X, axis=0)
            Y = np.concatenate(Y, axis=0)

        return X, Y

class Flip_Flop_Task(Task):
    """Generates data for the N-bit flip-flop task from Sussillo and Barak,
    2013."""

    def __init__(self, n_bit, p_flip, tau_task=1):

        super().__init__(n_bit, n_bit)

        self.p_flip = p_flip
        self.tau_task = tau_task

    def gen_dataset(self, N):

        N = N // self.tau_task
        
        if N == 0:
            return np.array([]), np.array([])

        probability = [self.p_flip / 2, 1 - self.p_flip, self.p_flip / 2]
        choices = [-1, 0, 1]
        X = np.random.choice(choices, size=(N, self.n_in), p=probability)
        X = np.tile(X, self.tau_task).reshape((self.tau_task*N, self.n_in))
        Y = X.copy()
        for k in range(int(np.ceil(np.log2(N)))):
            Y[2 ** k:] = np.sign(Y[2 ** k:] + Y[:-2 ** k] / 2)
        return X, Y







