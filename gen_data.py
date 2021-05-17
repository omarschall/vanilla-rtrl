#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 16:30:03 2018

@author: omarschall
"""

import numpy as np
import pickle
from pdb import set_trace

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

class Context_Dependent_Task(Task):
    
    def __init__(self, tasks, p_switch=0.5):
        
        self.tasks = tasks
        self.n_tasks = len(tasks)
        n_in = max([task.n_in for task in tasks]) + self.n_tasks
        n_out = max([task.n_out for task in tasks])
        
        super().__init__(n_in, n_out)
        
    def gen_dataset(self, N):
        
        print('hmm')
        
        

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

        with open('library/mnist.pkl', 'rb') as f:
            mnist = pickle.load(f)

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
    """Class for the N-bit flip-flop task.

    For n independent dimensions, an input stream of 0, -1 and 1 is provided,
    and the output is persistently either -1 or 1, flipping to the other only
    if the corresponding input is the opposite. Most inputs are 0, as
    specified by the "p_flip" parameter."""

    def __init__(self, n_bit, p_flip, tau_task=1, p_context_flip=None,
                 input_magnitudes=None,
                 x_dim_mask=[1,1,1], y_dim_mask=[1,1,1]):
        """Initiates an instance of the n-bit flip flop task by specifying the
        probability of a nonzero input and timescale of the task.

        Args:
            n_bit (int): The number of independent task dimensions.
            p_flip (float): The probability of an input being nonzero.
            tau_task (int): The factor by which we temporally "stretch" the task
                (similar to Add Task)."""

        self.n_bit = n_bit

        n_in = np.maximum(n_bit, 2)
        n_out = n_in
        
        if p_context_flip is not None:
            n_in += 1

        super().__init__(n_in, n_out)

        
        self.p_flip = p_flip
        self.tau_task = tau_task
        self.p_context_flip = p_context_flip
        self.input_magnitudes = input_magnitudes
        self.x_dim_mask = np.array(x_dim_mask)
        self.y_dim_mask = np.array(y_dim_mask)

    def gen_dataset(self, N):
        """Generates a dataset for the flip-flop task."""

        N = N // self.tau_task

        if N == 0:
            return np.array([]), np.array([])

        probability = [self.p_flip / 2, 1 - self.p_flip, self.p_flip / 2]
        choices = [-1, 0, 1]
        X = np.random.choice(choices, size=(N, self.n_bit), p=probability)
        X = np.tile(X, self.tau_task).reshape((self.tau_task*N, self.n_bit))
        Y = X.copy()
        for k in range(int(np.ceil(np.log2(N)))):
            Y[2 ** k:] = np.sign(Y[2 ** k:] + Y[:-2 ** k] / 2)
            
        if self.input_magnitudes is not None:
            mags = np.random.choice(self.input_magnitudes, size=X.shape)
            X = X * mags
            
        if self.n_bit == 1:
            X = np.tile(X, 2)
            Y = np.tile(Y, 2)
            
        if self.p_context_flip is not None:
            x_context = []
            init_context = np.random.choice([-1, 1])
            while len(x_context) < N:
                n_same_context = np.random.geometric(self.p_context_flip)
                x_context += ([init_context] * n_same_context)
                
                init_context *= -1
            x_context = np.array(x_context[:N]).reshape(-1, 1)
            
            X = np.concatenate([X, x_context], axis=1)
            
            X[np.where(x_context == -1), :-1] *= -1
            
        #Mask any dimensions
        X = X * self.x_dim_mask
        Y = Y * self.y_dim_mask
            
        return X, Y

class Sine_Wave(Task):
    """Class for the sine wave task.

    There are two input dimensions, one of which specifies whether the sine wave
    is "on" (1) or "off" (0). The second dimension specifies the period of
    the sine wave (in time steps) to be produced by the network."""

    def __init__(self, p_transition, periods, never_off=False, **kwargs):
        """Initializes an instance of sine wave task by specifying transition
        probability (between on and off states) and periods to sample from.

        Args:
            p_transition (float): The probability of switching between on and off
                modes.
            periods (list): The sine wave periods to sample from, by default
                uniformly.
            never_off (bool): If true, the is no "off" period, and the network
                simply switches from period to period.
        Keyword args:
            p_periods (list): Must be same length as periods, specifying probability
                for each sine wave period.
            amplitude (float): Amplitude of all sine waves, by default 0.1 if
                not specified.
            method (string): Must be either "random" or "regular", the former for
                transitions randomly sampled according to p_transition and the
                latter for deterministic transitions every 1 / p_transition
                time steps."""

        allowed_kwargs = {'p_periods', 'amplitude', 'method'}
        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError('Unexpected keyword argument '
                                'passed to Sine_Wave.__init__: ' + str(k))

        super().__init__(2, 2)

        self.p_transition = p_transition
        self.method = 'random'
        self.amplitude = 0.1
        self.periods = periods
        self.p_periods = np.ones_like(periods) / len(periods)
        self.never_off = never_off
        self.__dict__.update(kwargs)

    def gen_dataset(self, N):
        """Generates a dataset for the sine wave task."""

        X = np.zeros((N, 2))
        Y = np.zeros((N, 2))

        self.switch_cond = False

        active = False
        t = 0
        X[0, 0] = 1
        for i in range(1, N):

            if self.method == 'regular':
                if i % int(1 / self.p_transition) == 0:
                    self.switch_cond = True
            elif self.method == 'random':
                if np.random.rand() < self.p_transition:
                    self.switch_cond = True

            if self.switch_cond:

                t = 0

                if active and not self.never_off:
                    X[i, 0] = 1
                    X[i, 1] = 0
                    Y[i, :] = 0

                if not active or self.never_off:
                    X[i, 0] = np.random.choice(self.periods, p=self.p_periods)
                    X[i, 1] = 1
                    Y[i, 0] = self.amplitude * np.cos(2 * np.pi / X[i, 0] * t)
                    Y[i, 1] = self.amplitude * np.sin(2 * np.pi / X[i, 0] * t)

                active = not active

            else:

                t += 1
                X[i, :] = X[i - 1, :]
                theta = 2 * np.pi / X[i, 0] * t
                on_off = (active or self.never_off)
                Y[i, 0] = self.amplitude * np.cos(theta) * on_off
                Y[i, 1] = self.amplitude * np.sin(theta) * on_off

            self.switch_cond = False

        X[:, 0] = -np.log(X[:, 0])

        return X, Y

class Fixed_Point_Transition_Task(Task):
    """A task where a given number of states in output space is provided,
    along with a set of transition matrices as functions of possible inputs."""
    
    def __init__(self, states, T_dict, p_transition=0.05, deterministic=True,
                 delay=None):
        """Initializes an instance of the task by specifying the states
        (points in output space) to be reported, as well as a dictionary
        specifying the transition patterns for each input.
        
        Args:
            states (list): A list of length n_states, whose entries are umpy
                arrays with shape (n_out) that indicate in order the fixed
                points to be learned.
            T_dict (dictionary): A dictionary with keys 'input_{}'.format(k)
                for each possible input k, each pointing to a numpy array
                of shape (n_states, n_states) indicating transition probs.
            p_transition (float): A float indicating the probability of
                encountering a transition at each time point.
            deterministic (bool): A boolean indicating True if you know
                T_dict to be deterministic, which simplifies data gen.
            delay (int): An integer indicating the number of time steps to
                delay the indicated transition."""
                
        self.states = states
        self.n_states = len(states)
        self.T_dict = T_dict
        self.p_transition = p_transition
        self.deterministic = deterministic
        self.delay = delay
        
        super().__init__(len(T_dict.keys()), states[0].shape[0])
        
        #Assert dimensional consistency
        assert len(np.unique([state.shape[0] for state in states])) == 1
        for val in T_dict.values():
            assert self.n_states == val.shape[0]
            assert self.n_states == val.shape[1]
    
    def gen_dataset(self, N):
        
        #Indicate time steps where transition occurs
        p = [1 - self.p_transition, self.p_transition]
        sparsity_mask = np.random.choice([0, 1], size=N, p=p)
        sparsity_mask[0] = 0 #No transition at first time step
        T_times = np.where(sparsity_mask != 0)[0]
        
        #Sample particular transition types 
        I_X = np.random.randint(0, self.n_in, size=N) #* sparsity_mask
        X = np.eye(self.n_in)[I_X] #Turn into one-hot
        
        #apply sparsity
        I_X *= sparsity_mask
        X = (X.T * sparsity_mask).T
        
        #Set up initial state
        I_Y = [np.random.randint(self.n_states)]
        
        last_T_time = 0
        for i_T_time, T_time in enumerate(T_times):
            
            #Backfill all the last "same" states since last transition
            I_Y += ([I_Y[-1]] * (T_time - last_T_time - 1))
            
            #Access current transition and associated graph
            i_input = I_X[T_time]
            #set_trace()
            key = 'input_{}'.format(i_input)
            T_matrix = self.T_dict[key]
            
            if self.deterministic:
                I_Y.append(np.where(T_matrix[I_Y[-1]] > 0)[0][0])
            else:
                p_state = T_matrix[I_Y[-1]]
                I_Y.append(np.random.choice(list(range(self.n_states)),
                                            p=p_state))
                
            last_T_time = T_time
            
        #Final state
        
        I_Y += ([I_Y[-1]] * (len(I_X) - last_T_time - 1))
            
        Y = np.array([self.states[i_y] for i_y in I_Y])
        
        if self.delay is not None:
            Y = np.roll(Y, shift=self.delay, axis=0)
        
        return X, Y

class Multi_Task:
    """A class for online training in a multi-task setup. The class is initiated
    with a list of subclasses and a boolean indicating whether context inputs
    should be provided."""
    
    def __init__(self, tasks, context_input=False):
        """Initiate a multi-task object by specifying task *instances* to
        sample from.
        
        Args:
            tasks (list): A list whose entries are instances of Task
            context_input (bool): A boolean indicating whether to provide
                as an additional set of input dimensions an indication of
                which task is to be performed."""
        
        self.tasks = {i: tasks[i] for i in range(len(tasks))}
        self.context_input = context_input
        self.n_tasks = len(self.tasks)
        
        self.max_n_in = max([t.n_in for t in self.tasks.values()])
        self.max_n_out = max([t.n_out for t in self.tasks.values()])
        
        self.n_in = self.max_n_in + self.context_input * self.n_tasks
        self.n_out = self.max_n_out
        
        # if task_sample_method not in ['seq', 'random_choice']:
        #     raise ValueError('Invalid task_sample_method')
        # if task_duration_method not in ['uniform'. 'poisson']:
        #     raise ValueError('Invalid task_duration_method')
            
        # self.task_sample_method = task_sample_method
        # self.task_duration_method = task_duration_method
        
    def gen_train_dataset(self, N_train):
        """Generate a training dataset, which consists of a sampling of
        tasks."""
        
        if type(N_train)==int:
            Ns = [{'task_id': i,
                   'N': N_train // self.n_tasks} for i in range(self.n_tasks)]
        elif type(N_train)==list:
            Ns = N_train
        
        #Initialize total_data and task_marker with the first task.
        X, Y = self.tasks[Ns[0]['task_id']].gen_dataset(Ns[0]['N'])
        total_data = {'X': X, 'Y': Y}
        task_marker = [np.ones(Ns[0]['N']) * Ns[0]['task_id']]
        
        #Loop through the rest of the tasks and concatenate
        for i_block in range(1, len(Ns)):
            
            i_task = Ns[i_block]['task_id']
            task = self.tasks[i_task]
            N = Ns[i_block]['N']
            X, Y = task.gen_dataset(N)
            
            #Zero-pad lower-dimensional tasks in inputs and outputs
            if task.n_in < self.max_n_in:
                zero_pads = np.zeros((N, self.max_n_in - task.n_in))
                X = np.hstack([X, zero_pads])
                
            if task.n_out < self.max_n_out: 
                zero_pads = np.zeros((N, self.max_n_out - task.n_out))
                Y = np.hstack([Y, zero_pads])
            
            total_data['X'] = np.concatenate([total_data['X'], X], axis=0)
            total_data['Y'] = np.concatenate([total_data['Y'], Y], axis=0)
            
            task_marker.append(np.ones(N) * i_task)

        #Add task_marker to data
        total_data['task_marker'] = np.concatenate(task_marker).astype(np.int)
        
        #If specified, turn task_marker intoa one-hot and append to inputs
        if self.context_input:
            context = np.eye(self.n_tasks)[total_data['task_marker']]
            total_data['X'] = np.hstack([total_data['X'], context])
            
        return total_data
    
    def gen_data(self, N_train, N_test):
        
        data = {}
        
        data['train'] = self.gen_train_dataset(N_train)
        for i_task, task in zip(self.tasks.keys(), self.tasks.values()):
            
            X, Y = task.gen_dataset(N_test)
            key = 'test_{}'.format(i_task)
            data[key] = {'X': X, 'Y': Y}
            
            if self.context_input:
                task_id = (np.ones(N_test) * i_task).astype(np.int)
                context = np.eye(self.n_tasks)[task_id]
                data[key]['X'] = np.hstack([data[key]['X'], context])
            
        return data
            



