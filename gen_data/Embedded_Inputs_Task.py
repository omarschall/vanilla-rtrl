class Embedded_Inputs_Task:

    def __init__(self, task, W_embedding, f_embedding):

        self.task = task
        self.W_embedding = W_embedding
        self.f_embedding = f_embedding
        self.__dict__ = task.__dict__
        self.n_in = self.W_embedding.shape[1]
        self.probe_inputs = [self.embed_input(pi) for pi in task.probe_inputs]

    def gen_data(self, N_train, N_test):

        data = self.task.gen_data(N_train, N_test)

        for key in data:
            data[key]['X'] = self.embed_input(data[key]['X'])

        return data

    def embed_input(self, X):
        """Apply the instance's embedding function to a batch of data with shape
        (time, feature) or (feature)."""

        if len(X.shape) == 2:
            return self.f_embedding.f(self.W_embedding.dot(X.T)).T
        elif len(X.shape) == 1:
            return self.f_embedding.f(self.W_embedding.dot(X))