import torch
from .numpy_utils import torch_rnn_to_numpy_rnn

def train_torch_RNN_by_DNI(rnn, optimizer, A, A_optimizer, batched_data, batch_size, n_epochs,
                           L2_reg=0.0001, verbose=True, checkpoint_interval=None,
                           scheduler=None):
    n_batches = batched_data['train']['X'].shape[1] // batch_size
    T_trial = batched_data['train']['X'].shape[0]
    B = batch_size
    checkpoints = {}
    #A = torch.randn((rnn.n_h, rnn.n_h + rnn.n_out + 1), requires_grad=True)
    for i_epoch in range(n_epochs):
        for i_b in range(n_batches):

            X = torch.from_numpy(batched_data['train']['X'][:, i_b * B:(i_b + 1) * B, :]).to(torch.float32)
            Y = torch.from_numpy(batched_data['train']['Y'][:, i_b * B:(i_b + 1) * B, :]).to(torch.float32)

            state = torch.zeros((B, rnn.n_h))
            A_loss = 0
            loss = 0
            A_optimizer.zero_grad()
            optimizer.zero_grad()
            for p in rnn.parameters():
                p.grad = torch.zeros_like(p)
            A.grad = torch.zeros_like(A)
            for i_t_trial in range(T_trial):
                state, output = rnn(state, X[i_t_trial])
                loss += rnn.compute_loss(output, Y[i_t_trial])

                #Use A to estimate gradient
                a_tilde = torch.cat([state, Y[i_t_trial], torch.ones((B, 1))], dim=1)
                a_hat = torch.cat([rnn.state_prev, X[i_t_trial], torch.ones((B, 1))], dim=1)
                dL_da_hat = torch.einsum('bi,ij->bj', a_tilde, A)
                error = output - Y[i_t_trial]
                outer_grad = torch.einsum('bi,bj->bij',
                                          error, torch.cat([state, torch.ones((B, 1))], dim=1))
                rec_grads = torch.einsum('bi,bj->bij', dL_da_hat, a_hat)
                grads = [rec_grads[:,:,:rnn.n_h].mean(0), rec_grads[:,:,rnn.n_h:-1].mean(0), rec_grads[:,:,-1].mean(0)]
                grads += [outer_grad[:,:,:rnn.n_h].mean(0), outer_grad[:,:,-1].mean(0)]
                for i_p, p in enumerate(rnn.parameters()):
                    p.grad += grads[i_p]
                    if i_p in [0, 1, 3]:
                        p.grad += L2_reg * p.detach()
                loss += L2_reg * (torch.mean(torch.square(rnn.W_rec))
                                  + torch.mean(torch.square(rnn.W_in))
                                  + torch.mean(torch.square(rnn.W_out)))

                if i_t_trial > 0:
                    #Update A
                    pL_pa = torch.einsum('bi,ij->bj', error_prev, rnn.W_out)
                    J = rnn.alpha * torch.einsum('bi,ij->bij', rnn.activation_derivative(rnn.h), rnn.W_rec)
                    q_hat_bootstrap = pL_pa + torch.einsum('bi,bij->bj', dL_da_hat, J)
                    q_hat = torch.einsum('bi,ij->bj', a_tilde_prev, A)
                    A_loss += torch.mean(torch.square(q_hat_bootstrap - q_hat))
                    A.grad += torch.einsum('bi,bj->ij', a_tilde_prev, q_hat - q_hat_bootstrap) / batch_size
                error_prev = error
                a_tilde_prev = a_tilde

            optimizer.step()
            A_optimizer.step()
            if i_b % (n_batches // 10) == 0 and verbose:
                print(f'Epoch {i_epoch}, Batch {i_b}')
                print(f'Loss {loss.item()}')
                print(f'A Loss {A_loss.item()}')

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