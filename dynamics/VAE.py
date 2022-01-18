# from dynamics.dynamics_utils import *
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# class Encoder(nn.Module):
#
#     def __init__(self, input_dim, hidden_dims, z_dim):
#
#         super().__init__()
#
#         self.n_layers = len(hidden_dims)
#         self.linear_0 = nn.Linear(input_dim, hidden_dims[0])
#         for i in range(1, self.n_layers):
#             setattr(self, 'linear_{}'.format(i), nn.Linear(hidden_dims[i - 1],
#                                                            hidden_dims[i]))
#         self.mu = nn.Linear(hidden_dims[-1], z_dim)
#         self.var = nn.Linear(hidden_dims[-1], z_dim)
#
#     def forward(self, x):
#
#         # x is of shape [batch_size, input_dim]
#         for i in range(self.n_layers):
#             layer_linear = getattr(self, 'linear_{}'.format(i))
#             x = F.tanh(layer_linear(x))
#         # hidden is of shape [batch_size, hidden_dim]
#         z_mu = self.mu(x)
#         # z_mu is of shape [batch_size, latent_dim]
#         z_var = self.var(x)
#         # z_var is of shape [batch_size, latent_dim]
#
#         return z_mu, z_var
#
#
# class Decoder(nn.Module):
#
#     def __init__(self, z_dim, hidden_dims, output_dim):
#
#         super().__init__()
#
#         self.n_layers = len(hidden_dims)
#         self.linear_0 = nn.Linear(z_dim, hidden_dims[0])
#         for i in range(1, self.n_layers):
#             setattr(self, 'linear_{}'.format(i), nn.Linear(hidden_dims[i - 1],
#                                                            hidden_dims[i]))
#         self.out = nn.Linear(hidden_dims[-1], output_dim)
#
#     def forward(self, x):
#         # x is of shape [batch_size, latent_dim]
#
#         for i in range(self.n_layers):
#             layer_linear = getattr(self, 'linear_{}'.format(i))
#             x = F.tanh(layer_linear(x))
#         # hidden is of shape [batch_size, hidden_dim]
#
#         predicted = self.out(x)
#         # predicted is of shape [batch_size, output_dim]
#
#         return predicted
#
#
# class VAE(nn.Module):
#
#     def __init__(self, enc, dec):
#         super().__init__()
#
#         self.enc = enc
#         self.dec = dec
#
#     def forward(self, x):
#         # encode
#         z_mu, z_var = self.enc(x)
#
#         # sample from the distribution having latent parameters z_mu, z_var
#         # reparameterize
#         std = torch.exp(z_var / 2)
#         eps = torch.randn_like(std)
#         x_sample = eps.mul(std).add_(z_mu)
#
#         # decode
#         predicted = self.dec(x_sample)
#         return predicted, z_mu, z_var
#
# def train_VAE(checkpoint, data, T=20, n_epochs=5, lr=0.01,
#               latent_dim=10, encoder_dims=[256, 256],
#               decoder_dims=[256]):
#
#     #Generate data
#     A = get_test_sim_data(checkpoint, data)
#     train_data = torch.tensor(A.reshape((-1, T * checkpoint['rnn'].n_h)))
#
#     input_dim = train_data.shape[1]
#     #hidden_dim = 128
#
#     #set_trace()
#
#     #Define objects
#     encoder = Encoder(input_dim, encoder_dims, latent_dim)
#     decoder = Decoder(latent_dim, decoder_dims, input_dim)
#     model = VAE(encoder, decoder)
#
#     # optimizer
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#
#     #set_trace()
#
#     # set the train mode
#     model.train()
#
#     losses = []
#     recon_losses = []
#     kl_losses = []
#
#     for i_epoch in range(n_epochs):
#
#         # loss of the epoch
#         train_loss = 0
#         print('starting epoch {}'.format(i_epoch))
#         train_iterator = DataLoader(TensorDataset(train_data),
#                                     batch_size=20,
#                                     shuffle=True)
#
#         #set_trace()
#
#         for i, x in enumerate(train_iterator):
#             # reshape the data into [batch_size, 784]
#             x = x[0].view(-1, input_dim).type(torch.FloatTensor)
#
#             #set_trace()
#
#             # update the gradients to zero
#             optimizer.zero_grad()
#
#             # forward pass
#             x_sample, z_mu, z_var = model(x)
#
#             # reconstruction loss
#             recon_loss = F.mse_loss(x_sample, x, size_average=False)
#
#             # kl divergence loss
#             kl_loss = 0.5 * torch.sum(torch.exp(z_var) +
#                                       z_mu ** 2 - 1.0 - z_var)
#
#             # total loss
#             loss = recon_loss + kl_loss
#
#             recon_losses.append(recon_loss.detach().numpy())
#             kl_losses.append(kl_loss.detach().numpy())
#             losses.append(loss.detach().numpy())
#
#             # backward pass
#             loss.backward()
#             train_loss += loss.item()
#
#             # update the weights
#             optimizer.step()
#
#     checkpoint['VAE'] = model
#     checkpoint['VAE_T'] = T
#     checkpoint['VAE_train_losses'] = losses
#     checkpoint['VAE_kl_losses'] = kl_losses
#     checkpoint['VAE_recon_losses'] = recon_losses
#
# def sample_from_VAE(checkpoint):
#
#     model = checkpoint['VAE']
#     T = checkpoint['VAE_T']
#     # sample and generate a image
#
#     z = torch.randn(1, model.enc.mu.out_features)
#
#     # run only the decoder
#     reconstructed_traj = model.dec(z)
#     traj = reconstructed_traj.view(T, -1).data
#
#     return traj
#
# def test_vae(model_checkpoint, data, test_checkpoint=None):
#
#     model = model_checkpoint['VAE']
#     n_h = model_checkpoint['rnn'].n_h
#     T = model_checkpoint['VAE_T']
#
#     if test_checkpoint is None:
#         A = get_test_sim_data(model_checkpoint, data)
#
#     else:
#         A = get_test_sim_data(test_checkpoint, data)
#
#     test_data = torch.tensor(A.reshape((-1, T * n_h)))
#
#     test_iterator = DataLoader(TensorDataset(test_data),
#                                batch_size=A.shape[0],
#                                shuffle=False)
#     input_dim = test_data.shape[1]
#     # set the evaluation mode
#     model.eval()
#
#     # test loss for the data
#     test_loss = 0
#
#     with torch.no_grad():
#         for i, x in enumerate(test_iterator):
#             # reshape the data
#             x = x[0].view(-1, input_dim).type(torch.FloatTensor)
#
#             # forward pass
#             x_sample, z_mu, z_var = model(x)
#
#             # reconstruction loss
#             recon_loss = F.mse_loss(x_sample, x, size_average=False)
#
#             # kl divergence loss
#             kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1.0 - z_var)
#
#             # total loss
#             loss = recon_loss + kl_loss
#             test_loss += loss.item()
#
#     return test_loss
#
#
#
#
