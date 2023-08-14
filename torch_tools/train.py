import torch
from .numpy_utils import torch_rnn_to_numpy_rnn

def train_torch_RNN(rnn, optimizer, batched_data, batch_size, n_epochs,
                    L2_reg=0.0001, verbose=True, checkpoint_interval=None,
                    scheduler=None):
    n_batches = batched_data['train']['X'].shape[1] // batch_size
    T_trial = batched_data['train']['X'].shape[0]
    B = batch_size
    checkpoints = {}
    for i_epoch in range(n_epochs):
        for i_b in range(n_batches):

            X = torch.from_numpy(batched_data['train']['X'][:, i_b * B:(i_b + 1) * B, :]).to(torch.float32)
            Y = torch.from_numpy(batched_data['train']['Y'][:, i_b * B:(i_b + 1) * B, :]).to(torch.float32)

            state = torch.zeros((B, rnn.n_h))
            loss = 0
            for i_t_trial in range(T_trial):
                state, output = rnn(state, X[i_t_trial])
                loss += rnn.compute_loss(output, Y[i_t_trial])

            loss += L2_reg * (torch.mean(torch.square(rnn.W_rec))
                              + torch.mean(torch.square(rnn.W_in))
                              + torch.mean(torch.square(rnn.W_out)))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i_b % (n_batches // 10) == 0 and verbose:
                print(f'Epoch {i_epoch}, Batch {i_b}')
                print(f'Loss {loss.item()}')

            if checkpoint_interval is not None:
                if i_b % checkpoint_interval == 0:
                    i_t = i_b + i_epoch * n_batches
                    checkpoints[i_t] = {'rnn': torch_rnn_to_numpy_rnn(rnn),
                                        'i_t': i_t}
        if scheduler is not None:
            scheduler.step()
    checkpoints['final'] = {'rnn': torch_rnn_to_numpy_rnn(rnn),
                            'i_t': n_epochs * n_batches}

    return checkpoints

def train_torch_RNN_by_RFLO(rnn, optimizer, batched_data, batch_size, n_epochs,
                    L2_reg=0.0001, verbose=True, checkpoint_interval=None,
                    scheduler=None):
    n_batches = batched_data['train']['X'].shape[1] // batch_size
    T_trial = batched_data['train']['X'].shape[0]
    B = batch_size
    checkpoints = {}
    E_trace = torch.zeros((B, rnn.n_h, rnn.n_h + rnn.n_in + 1))
    for i_epoch in range(n_epochs):
        for i_b in range(n_batches):

            X = torch.from_numpy(batched_data['train']['X'][:, i_b * B:(i_b + 1) * B, :]).to(torch.float32)
            Y = torch.from_numpy(batched_data['train']['Y'][:, i_b * B:(i_b + 1) * B, :]).to(torch.float32)

            state = torch.zeros((B, rnn.n_h))
            loss = 0
            for p in rnn.parameters():
                p.grad = torch.zeros_like(p)
            for i_t_trial in range(T_trial):
                state, output = rnn(state, X[i_t_trial])
                loss += rnn.compute_loss(output, Y[i_t_trial])
                a_hat = torch.cat([rnn.state_prev, X[i_t_trial], torch.ones((B, 1))], dim=1)
                E_immediate = torch.einsum('bi,bj->bij', rnn.activation_derivative(rnn.h), a_hat)
                E_trace = (1 - rnn.alpha) * E_trace + rnn.alpha * E_immediate
                for p in rnn.parameters():
                    p.grad += torch.einsum('bij,bj->bi', E_trace, rnn.dL_dh)
            loss += L2_reg * (torch.mean(torch.square(rnn.W_rec))
                              + torch.mean(torch.square(rnn.W_in))
                              + torch.mean(torch.square(rnn.W_out)))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i_b % (n_batches // 10) == 0 and verbose:
                print(f'Epoch {i_epoch}, Batch {i_b}')
                print(f'Loss {loss.item()}')

            if checkpoint_interval is not None:
                if i_b % checkpoint_interval == 0:
                    i_t = i_b + i_epoch * n_batches
                    checkpoints[i_t] = {'rnn': torch_rnn_to_numpy_rnn(rnn),
                                        'i_t': i_t}
        if scheduler is not None:
            scheduler.step()
    checkpoints['final'] = {'rnn': torch_rnn_to_numpy_rnn(rnn),
                            'i_t': n_epochs * n_batches}

    return checkpoints