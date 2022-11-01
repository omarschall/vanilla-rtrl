from copy import copy, deepcopy
import time
from utils import *
import numpy as np

class Simulation:
    """Simulates an RNN for a provided set of inputs and training procedures.

    By default, all variables are overwritten at each time step, but the user
    can specify which variables to track with 'monitors'.

    Has two types of attributes that are provided either upon init or
    when calling self.run. The distinction matters because init attributes
    should carry over into test runs, whereas run attributes will likely be
    different in train and test runs. Details given in __init__ and run
    docstrings."""

    def __init__(self, rnn, allowed_kwargs_=set(), **kwargs):
        """Initialzes a simulation.Simulation object by specifying the
        attributes that will apply to both train and test instances.

        Args:
            rnn (network.RNN): The specific RNN instance being simulated.
            allowed_kwargs_ (set): Custom allowed keyword args for development
                of subclasses of simulation that need additional specification.
            time_steps_per_trial (int): Number of time steps in a trial, if
                task has a trial structure. Leave empty if task runs
                continuously.
            trial_mask (numpy array): Array of shape (time_steps_per_trial)
                that scales the loss at each time step within a trial.
            reset_sigma (float): Standard deviation of RNN initial state at
                start of each trial. Leave empty if RNN state should carry over
                from last state of previous trial.
            i_job (int): An integer indexing the job that this simulation
                corresponds to if submitting a batch job to cluster.
            save_dir (string): Path indicating where to save intermediate
                results, if desired."""

        allowed_kwargs = {'time_steps_per_trial',
                          'reset_sigma', 'trial_mask',
                          'i_job', 'save_dir'}.union(allowed_kwargs_)
        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError('Unexpected keyword argument '
                                'passed to Simulation.__init__: ' + str(k))
        self.rnn = rnn
        self.__dict__.update(kwargs)

        #Set to None all unspecified attributes
        for attr in allowed_kwargs:
            if not hasattr(self, attr):
                setattr(self, attr, None)

        if self.rnn.reset_sigma is not None:
            self.reset_sigma = self.rnn.reset_sigma

    def run(self, data, mode='train', monitors=[], **kwargs):
        """Runs the network forward as many time steps as given by data.

        Can be run in either 'train' or 'test' mode, which specifies whether
        the network parameters are updated or not. In 'train' case, a
        learning algorithm (learn_alg) and optimizer (optimizer) must be
        specified to calculate gradients and apply them to the network,
        respectively.

        Args:
            data (dict): A dict containing two keys, 'train' and 'test',
                each of which points to a dict containing keys 'X' and 'Y',
                providing numpy arrays of inputs and labels, respectively,
                of shape (T, n_in) or (T, n_out).
            mode (string): A string that must be either 'train' or 'test'
                which indicates which dict in data to use and whether to
                update network parameters while running.
            monitors (list): A list of strings, such that if a string matches
                an attribute of any relevant object in the simluation,
                including the simulation itself, the network, the optimizer,
                or the learning algorithm, that attribute's value is stored
                at each time step. If there is a '-' (hyphen) between the name
                and either 'radius' or 'norm', then the spectral radius or
                norm, respectively, of that object is stored instead.
            learn_alg (learning_algorithms.Learning_Algorithm): The instance
                of a learning algorithm used to calculate the gradients.
            optimizer (optimizers.Optimizer): The instance of an optimizer
                used to update the network using gradients computed by
                learn_alg.
            update_interval (int): Number of time steps between each parameter
                update.
            a_initial (numpy array): An array of shape (rnn.n_hidden) that
                specifies the initial state of the network when running. If
                not specified, the default initialization practice is inherited
                from the RNN.
            sigma (float): Specifies standard deviation of white noise to add
                to the network pre-activations at each time step.
            comp_algs (list): A list of instances of Learning_Algorithm
                specified to passively run in parallel with learn_alg to enable
                comparisons between the algorithms.
            verbose (bool): Boolean that indicates whether to print progress
                reports during the simulation.
            report_interval (int): Number of time steps between reports, if
                verbose. Default is 1/10 the number of total time steps, for 10
                total progress reports.
            report_accuracy (bool): Boolean that indicates whether to run test
                simulation during each progress report to report classification
                accuracy, as defined by the fraction of test time steps where
                the argmax of the outputs matches that of the labels.
            report_loss (bool): Same as report_accuracy but with test loss. If
                both are False, no test simulation is run.
            best_model_interval (int): Number of time steps between running
                test simulations and saving the model if it has the lowest
                yet validation loss.
            checkpoint_interval (int): Number of time steps between saving
                rnn, learn_alg, optimizer, and i_t so that training can be
                reproduced.
            'N_Duncker_data (int): Number of past data points to use for
                calculating Duncker projections during a task switch."""

        allowed_kwargs = {'learn_alg', 'optimizer', 'a_initial', 'sigma',
                          'update_interval', 'comp_algs', 'verbose', 'print',
                          'test_current_task',
                          'report_interval', 'report_accuracy', 'report_loss',
                          'best_model_interval', 'checkpoint_interval',
                          'overwrite_checkpoints', 'i_start', 'i_end',
                          'checkpoint_learn_alg', 'checkpoint_optimizer'}
        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError('Unexpected keyword argument '
                                'passed to Simulation.run: ' + str(k))

        ### --- Set new object attributes for run --- ###

        #Create new pointers for conveneince
        self.mode = mode
        self.monitors = monitors
        self.x_inputs = data[mode]['X']
        self.y_labels = data[mode]['Y']
        if 'trial_type' in data[mode].keys():
            self.trial_type = data[mode]['trial_type']
        if 'task_marker' in data[mode].keys():
            self.task_marker = data[mode]['task_marker']
        if 'trial_switch' in data[mode].keys():
            self.trial_switch = data[mode]['trial_switch']
        if 'loss_mask' in data[mode].keys():
            self.loss_mask = data[mode]['loss_mask']
        self.total_time_steps = self.x_inputs.shape[0]

        #Set defaults
        self.verbose = True
        self.print = False
        self.test_current_task = True
        self.report_accuracy = False
        self.report_loss = False
        self.comp_algs = []
        self.report_interval = max(self.total_time_steps//10, 1)
        self.update_interval = 1
        self.i_start = 0
        self.i_trial = 0
        self.i_end = self.total_time_steps
        self.sigma = 0
        self.checkpoint_learn_alg = False
        self.checkpoint_optimizer = False

        #Overwrite defaults with any provided keyword args
        self.__dict__.update(kwargs)

        #Set to None all unspecified attributes
        for attr in allowed_kwargs:
            if not hasattr(self, attr):
                setattr(self, attr, None)

        if self.reset_sigma is not None:
            self.rnn.reset_sigma = self.reset_sigma

        ### --- Pre-run housekeeping --- ###

        self.initialize_run()

        for i_t in range(self.i_start, self.i_end):

            self.i_t = i_t

            ### --- Reset model if there is a trial structure --- ###

            if self.trial_switch is not None:
                self.trial_structure()

            ### --- Run network forwards and get error --- ###

            self.forward_pass(self.x_inputs[i_t],
                              self.y_labels[i_t])

            ### --- Update parameters if in 'train' mode --- ###

            if self.mode == 'train':
                self.train_step()

            ### --- Clean up --- ###

            self.end_time_step(data)

        #At end of run, convert monitor lists into numpy arrays
        self.monitors_to_arrays()

        #Checkpoint final model
        self.checkpoint_model()

        #Delete data to save space
        del(self.x_inputs)
        del(self.y_labels)
        if 'task_marker' in data[mode].keys():
            del(self.task_marker)

    def initialize_run(self):
        """Initializes a few variables before the time loop."""

        #Initial best validation loss is infinite
        self.best_val_loss = np.inf

        #Set up checkpoints dict if doesn't already exist from previous run
        if not hasattr(self, 'checkpoints'):
            self.checkpoints = {}

        #Initialize rec_grads_dicts
        if self.mode == 'train':
            self.algs = [self.learn_alg] + self.comp_algs
            self.rec_grads_dict = {alg.name:[] for alg in self.algs}
            self.T_lag = 0
            for alg in self.algs:
                try:
                    if alg.T_truncation > self.T_lag:
                        self.T_lag = alg.T_truncation
                except AttributeError:
                    pass

        #Initialize monitors
        self.mons = {mon:[] for mon in self.monitors}
        #Make all relevant algorithms attributes of self
        if self.mode == 'train':
            for comp_alg in self.comp_algs:
                setattr(self, comp_alg.name, comp_alg)
            setattr(self, self.learn_alg.name, self.learn_alg)

        #Set a random initial state of the network
        if self.a_initial is not None:
            self.rnn.reset_network(a=self.a_initial)

        #To avoid errors, initialize "previous"
        #inputs/labels as the first inputs/labels
        self.rnn.x_prev = self.x_inputs[0]
        self.rnn.y_prev = self.y_labels[0]

        #Track computation time
        self.start_time = time.time()

    def trial_structure(self):
        """Resets learning algorithm and/or network state between trials."""

        if self.trial_switch[self.i_t] == 1:
            self.i_trial += 1
            try:
                self.rnn.trial_type = self.trial_type[self.i_t]
            except TypeError:
                pass
            if self.reset_sigma is not None:
                self.rnn.reset_network(sigma=self.reset_sigma)
                try:
                    self.learn_alg.reset_learning()
                except AttributeError:
                    pass

    def forward_pass(self, x, y):
        """Runs network forward, computes immediate losses and errors."""

        #Pointer for convenience
        rnn = self.rnn

        #Pass data to network
        rnn.x = x
        rnn.y = y

        #Run network forwards and get predictions
        rnn.next_state(rnn.x, sigma=self.sigma)
        rnn.z_out()

        #Compare outputs with labels, get immediate loss and errors
        rnn.y_hat = rnn.output.f(rnn.z)
        rnn.loss_ = rnn.loss.f(rnn.z, rnn.y)
        rnn.error = rnn.loss.f_prime(rnn.z, rnn.y)

        #Re-scale losses and errors if trial structure is provided
        if self.loss_mask is not None:
            #Is this clunky numpy code or the slickeset use of numpy broadcasting
            #rules maybe ever? You be the judge. Effortless translation into
            #dimension-wise loss if that's a thing you care about.
            rnn.loss_ *= self.loss_mask[self.i_t]
            rnn.error *= self.loss_mask[self.i_t]

    def train_step(self):
        """Uses self.learn_alg to calculate gradients and self.optimizer to
        apply them to self.rnn. Also calculates gradients from comparison
        algorithms."""

        #Pointer for convenience
        rnn = self.rnn

        ### --- Continual learning --- ###

        if self.learn_alg.CL_method is not None and self.i_t > 0:
            self.learn_alg.CL_method.mini_update(self)
            if self.task_marker[self.i_t] != self.task_marker[self.i_t - 1]:
                self.learn_alg.CL_method.task_switch_update(self)

        ### --- Calculate gradients --- ###

        #Update learn_alg variables and get gradients
        self.learn_alg.update_learning_vars()
        self.grads_list = self.learn_alg()

        ### --- Calculate gradients for comparison algorithms --- ###

        if len(self.comp_algs) > 0:
            self.compare_algorithms()

        ### --- Pass gradients to optimizer --- ###

        #Only update on schedule (default update_interval=1)
        if self.i_t%self.update_interval == 0:
            #Get updated parameters
            rnn.params = self.optimizer.get_updated_params(rnn.params,
                                                           self.grads_list)
            rnn.W_rec, rnn.W_in, rnn.b_rec, rnn.W_out, rnn.b_out = rnn.params

    def end_time_step(self, data):
        """Cleans up after each time step in the time loop."""

        #Compute spectral radii if desired
        self.get_radii_and_norms()

        #Monitor relevant variables
        self.update_monitors()

        #Evaluate model and save if performance is best
        if self.best_model_interval is not None and self.mode == 'train':
            if self.i_t % self.best_model_interval == 0:
                self.save_best_model(data)

        if self.checkpoint_interval is not None and self.mode == 'train':
            if type(self.checkpoint_interval) is int:
                if self.i_t % self.checkpoint_interval == 0:
                    self.checkpoint_model()
            if type(self.checkpoint_interval) is list:
                if self.i_t in self.checkpoint_interval:
                    self.checkpoint_model()

        #Make report if conditions are met
        if (self.i_t % self.report_interval == 0 and
            self.i_t > 0 and
            self.verbose):
            self.report_progress(data)

        #Current inputs/labels become previous inputs/labels
        self.rnn.x_prev = self.rnn.x.copy()
        self.rnn.y_prev = self.rnn.y.copy()

    def report_progress(self, data):
        """"Reports progress at specified interval, including test run
        performance if specified."""

        progress = np.round((self.i_t/self.total_time_steps)*100, 2)
        time_elapsed = np.round(time.time() - self.start_time, 1)

        summary = '\rProgress: {}% complete \nTime Elapsed: {}s \n'
        mode = 'test'

        if 'task_marker' in data['train'].keys() and self.test_current_task:
            mode += '_{}'.format(data['train']['task_marker'][self.i_t])

        if 'rnn.loss_' in self.mons.keys():
            interval = self.report_interval
            avg_loss = sum(self.mons['rnn.loss_'][-interval:])/interval
            loss = 'Average loss: {} \n'.format(avg_loss)
            summary += loss

        if self.report_accuracy or self.report_loss:
            test_sim = self.get_test_sim()
            test_sim.run(data, mode=mode,
                         monitors=['rnn.y_hat', 'rnn.loss_'],
                         verbose=False,
                         a_initial=np.copy(self.rnn.a))
            if self.report_accuracy:
                acc = classification_accuracy(data, test_sim.mons['rnn.y_hat'])
                accuracy = 'Test accuracy: {} \n'.format(acc)
                summary += accuracy
            if self.report_loss:
                test_loss = np.mean(test_sim.mons['rnn.loss_'])
                loss_summary = 'Test loss: {} \n'.format(test_loss)
                summary += loss_summary
                self.test_loss = test_loss

        if self.print:
            print(summary.format(progress, time_elapsed))

    def update_monitors(self):
        """Loops through the monitor keys and appends current value of any
        object's attribute found."""

        for key in self.mons:
            try:
                self.mons[key].append(rgetattr(self, key))
            except AttributeError:
                pass

    def monitors_to_arrays(self):
        """Recasts monitors (lists by default) as numpy arrays for ease of use
        after running."""

        for key in self.mons:
            try:
                self.mons[key] = np.array(self.mons[key])
            except ValueError:
                pass

    def get_radii_and_norms(self):
        """Calculates the spectral radii and/or norms of any monitor keys
        where this is specified."""

        for feature, func in zip(['radius', 'norm'],
                                 [get_spectral_radius, norm]):
            for key in self.mons:
                if feature in key:
                    attr = key.split('-')[0]
                    self.mons[key].append(func(rgetattr(self, attr)))

    def save_best_model(self, data):
        """Runs a test simulation, compares loss to current best model, and
        replaces the best model if the test loss is lower than previous lowest
        test loss."""

        val_sim = self.get_test_sim()
        val_sim.run(data, mode='test',
                    monitors=['rnn.y_hat', 'rnn.loss_'],
                    verbose=False)
        val_loss = np.mean(val_sim.mons['rnn.loss_'])

        if val_loss < self.best_val_loss:
            self.best_rnn = val_sim.rnn
            self.best_val_loss = val_loss

    def checkpoint_model(self):
        """Creates copies of all relevant objects for reproducing training
        trajectory."""

        checkpoint = {'rnn': deepcopy(self.rnn),
                      'i_t': copy(self.i_t)}

        if self.checkpoint_learn_alg:
            checkpoint['learn_alg'] = deepcopy(self.learn_alg)
        if self.checkpoint_optimizer:
            checkpoint['optimizer'] = deepcopy(self.optimizer)
        if (self.i_t not in self.checkpoints.keys() or
            self.overwrite_checkpoints):
            self.checkpoints[self.i_t] = checkpoint

    def get_test_sim(self):
        """Creates what is effectively a copy of the current simulation, but
        saving on memory by omitting monitors or other large attributes."""

        sim = Simulation(deepcopy(self.rnn),
                         time_steps_per_trial=self.time_steps_per_trial,
                         trial_mask=self.trial_mask,
                         reset_sigma=self.reset_sigma,
                         i_job=self.i_job,
                         save_dir=self.save_dir)
        return sim

    def compare_algorithms(self):
        """Computes alignment matrix for different learning algorithms run
        in parallel."""

        #Update learning variables for the algorithms *not* being used to train
        #the network
        for i_alg, alg in enumerate(self.algs):
            if i_alg > 0: #Only the comparison algorithms
                alg.update_learning_vars()
                alg()
            key = alg.name
            #Store the rec_grad array for each algorithm in a list
            self.rec_grads_dict[key].append(alg.rec_grads)
        #Get array of gradient alignments
        if 'alignment_matrix' in self.mons.keys():
            n_algs = len(self.algs)
            self.alignment_matrix = np.zeros((n_algs, n_algs))
            self.alignment_weights = np.zeros((n_algs, n_algs))
            for i, key_i in enumerate(self.rec_grads_dict):
                for j, key_j in enumerate(self.rec_grads_dict):

                    #For comparison with Future_BPTT, must lag gradients by
                    #the truncation horizon
                    if 'F-BPTT' in key_i:
                        i_index = -1
                    else:
                        i_index = 0
                    if 'F-BPTT' in key_j:
                        j_index = -1
                    else:
                        j_index = 0

                    #Store normalized dot product for each pair of algorithms
                    #in the alignment matrix and norm product in alignment
                    #strength matrix.
                    g_i = self.rec_grads_dict[key_i][i_index]
                    g_j = self.rec_grads_dict[key_j][j_index]
                    alignment = normalized_dot_product(g_i, g_j)
                    self.alignment_matrix[i, j] = alignment
                    self.alignment_weights[i, j] = norm(g_i) * norm(g_j)

        #Keep each list (for each algorithm) of rec_grads only as long as the
        #truncation horizon by deleting the oldest one.
        for key in self.rec_grads_dict:
            if len(self.rec_grads_dict[key]) >= self.T_lag:
                del(self.rec_grads_dict[key][0])

    def resume_sim_at_checkpoint(self, data, i_checkpoint, N=None,
                                 overwrite_checkpoints=False,
                                 new_learn_alg=None,
                                 new_optimizer=None,
                                 new_rnn=None,
                                 auto_resample=True,
                                 **kwargs):
        """Restarts the simulation at a given checkpoint. The state of the RNN,
        learning algorithm, and optimizer are all returned to that point."""

        checkpoint = self.checkpoints[i_checkpoint]

        if new_rnn is None:
            rnn = deepcopy(checkpoint['rnn'])
        else:
            rnn = new_rnn
        if new_learn_alg is None:
            learn_alg = deepcopy(checkpoint['learn_alg'])
        else:
            learn_alg = new_learn_alg
        if new_optimizer is None:
            optimizer = deepcopy(checkpoint['optimizer'])
        else:
            optimizer = new_optimizer

        if auto_resample:

            i_checkpoints = sorted(self.checkpoints.keys())
            j = i_checkpoints.index(i_checkpoint) + 1
            N = i_checkpoints[j] - i_checkpoints[j-1]
            checkpoint_interval = N // 10

            self.rnn = rnn
            self.run(data, learn_alg=learn_alg, optimizer=optimizer,
                     checkpoint_interval=checkpoint_interval,
                     i_start=i_checkpoint + 1,
                     i_end=i_checkpoint + N,
                     overwrite_checkpoints=overwrite_checkpoints,
                     **kwargs)
        else:
            self.rnn = rnn
            self.run(data, learn_alg=learn_alg, optimizer=optimizer,
                     i_start=i_checkpoint + 1,
                     i_end=i_checkpoint + N,
                     checkpoint_interval=N,
                     overwrite_checkpoints=overwrite_checkpoints,
                     **kwargs)

































