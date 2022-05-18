import numpy as np

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
        try:
            self.probe_inputs = tasks[-1].probe_inputs
        except AttributeError:
            pass
        self.context_input = context_input
        self.n_tasks = len(self.tasks)

        self.max_n_in = max([t.n_in for t in self.tasks.values()])
        self.max_n_out = max([t.n_out for t in self.tasks.values()])

        self.n_in = self.max_n_in + self.context_input * self.n_tasks
        self.n_out = self.max_n_out

    def gen_train_dataset(self, N_train):
        """Generate a training dataset, which consists of a sampling of
        tasks."""

        if type(N_train)==int:
            Ns = [{'task_id': i,
                   'N': N_train // self.n_tasks} for i in range(self.n_tasks)]
        elif type(N_train)==list:
            Ns = N_train

        #Initialize total_data and task_marker with the first task.
        X, Y, trial_type, trial_switch, loss_mask = self.tasks[Ns[0]['task_id']].gen_dataset(Ns[0]['N'])
        total_data = {'X': X, 'Y': Y, 'trial_type': trial_type,
                      'trial_switch': trial_switch, 'loss_mask': loss_mask}
        task_marker = [np.ones(Ns[0]['N']) * Ns[0]['task_id']]

        #Loop through the rest of the tasks and concatenate
        for i_block in range(1, len(Ns)):

            i_task = Ns[i_block]['task_id']
            task = self.tasks[i_task]
            N = Ns[i_block]['N']
            X, Y, trial_type, trial_switch, loss_mask = task.gen_dataset(N)

            #Zero-pad lower-dimensional tasks in inputs and outputs
            if task.n_in < self.max_n_in:
                zero_pads = np.zeros((N, self.max_n_in - task.n_in))
                X = np.hstack([X, zero_pads])

            if task.n_out < self.max_n_out:
                zero_pads = np.zeros((N, self.max_n_out - task.n_out))
                Y = np.hstack([Y, zero_pads])

            total_data['X'] = np.concatenate([total_data['X'], X], axis=0)
            total_data['Y'] = np.concatenate([total_data['Y'], Y], axis=0)
            if trial_type is not None:
                total_data['trial_type'] = np.concatenate([total_data['trial_type'], trial_type], axis=0)
            if trial_switch is not None:
                total_data['trial_switch'] = np.concatenate([total_data['trial_switch'], trial_switch], axis=0)
            if loss_mask is not None:
                total_data['loss_mask'] = np.concatenate([total_data['loss_mask'], loss_mask], axis=0)

            task_marker.append(np.ones(N) * i_task)

        #Add task_marker to data
        total_data['task_marker'] = np.concatenate(task_marker).astype(np.int)

        #If specified, turn task_marker into a one-hot and append to inputs
        if self.context_input:
            context = np.eye(self.n_tasks)[total_data['task_marker']]
            total_data['X'] = np.hstack([total_data['X'], context])

        return total_data

    def gen_data(self, N_train, N_test):

        data = {}

        data['train'] = self.gen_train_dataset(N_train)
        for i_task, task in zip(self.tasks.keys(), self.tasks.values()):

            X, Y, trial_type, trial_switch, loss_mask = task.gen_dataset(N_test)
            key = 'test_{}'.format(i_task)
            data[key] = {'X': X, 'Y': Y, 'trial_type': trial_type,
                         'trial_switch': trial_switch, 'loss_mask': loss_mask}

            if self.context_input:
                task_id = (np.ones(N_test) * i_task).astype(np.int)
                context = np.eye(self.n_tasks)[task_id]
                data[key]['X'] = np.hstack([data[key]['X'], context])

            if i_task == len(self.tasks.keys()) - 1:
                data['test'] = data[key]

        return data
