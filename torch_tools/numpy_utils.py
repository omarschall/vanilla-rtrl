from functions import tanh, relu, identity, mean_squared_error
from core import RNN

def torch_rnn_to_numpy_rnn(torch_rnn, output=identity, loss=mean_squared_error, reset_sigma=None):
    """Translate RNN object for training in torch to instance of core.RNN class."""

    if torch_rnn.activation_name == 'tanh':
        activation = tanh
    elif torch_rnn.activation_name == 'relu':
        activation = relu
    rnn = RNN(torch_rnn.W_in.detach().numpy(),
              torch_rnn.W_rec.detach().numpy(),
              torch_rnn.W_out.detach().numpy(),
              torch_rnn.b_rec.detach().numpy(),
              torch_rnn.b_out.detach().numpy(),
              activation=activation, alpha=torch_rnn.alpha, output=output,
              loss=loss, reset_sigma=reset_sigma)

    return rnn


def add_batch_dimension_to_data(data, T_trial):
    """Reshape task data into [time in trial, batch, dim] from time-concatenated trials."""

    n_in = data['train']['X'].shape[-1]
    n_out = data['train']['Y'].shape[-1]

    train_inputs = data['train']['X'].reshape(T_trial, -1, n_in, order='F')
    train_labels = data['train']['Y'].reshape(T_trial, -1, n_out, order='F')
    test_inputs = data['test']['X'].reshape(T_trial, -1, n_in, order='F')
    test_labels = data['test']['Y'].reshape(T_trial, -1, n_out, order='F')

    batched_data = {'train': {'X': train_inputs, 'Y': train_labels},
                    'test': {'X': test_inputs, 'Y': test_labels}}

    return batched_data

class Empty_Simulation:
    """Empty simulation object that will help downstream analyses easier, mainly
    just a space to hold a checkpointd dict."""

    def __init__(self):
        pass