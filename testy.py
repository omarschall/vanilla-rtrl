import sys, os, pickle
sys.path.append('/vanilla-rtrl/')
from cluster import *
from continual_learning import *
from core import *
from dynamics import *
from functions import *
from gen_data import *
from learning_algorithms import *
from optimizers import *
from plotting import *
from wrappers import *
import matplotlib.pyplot as plt

task = Flip_Flop_Task(3, 0.05, input_magnitudes=None)
#n_bit (int): The number of independent task dimensions, meaning 3 dimensions.
#p_flip (float): The probability of an input being nonzero, meaning most inputs are 0.
N_train = 200000
N_test = 2000
checkpoint_interval = None
sigma = 0

data = task.gen_data(N_train, N_test)

#n_in = np.maximum(n_bit, 2) = n_out
#3 dimensions
n_in = task.n_in
n_hidden = 32
n_out = task.n_out

#Initialize weights and biases

Wz_in = np.random.normal(0, np.sqrt(1/(n_in)), (n_hidden, n_in))
Wr_in = np.random.normal(0, np.sqrt(1/(n_in)), (n_hidden, n_in))
Wh_in = np.random.normal(0, np.sqrt(1/(n_in)), (n_hidden, n_in))
Wz_rec = np.random.normal(0, np.sqrt(1/n_hidden), (n_hidden, n_hidden))
Wr_rec = np.random.normal(0, np.sqrt(1/n_hidden), (n_hidden, n_hidden))
Wh_rec = np.random.normal(0, np.sqrt(1/n_hidden), (n_hidden, n_hidden))
W_out = np.random.normal(0, np.sqrt(1/(n_hidden)), (n_out, n_hidden))

bz_rec = np.zeros(n_hidden)
br_rec = np.zeros(n_hidden)
bh_rec = np.zeros(n_hidden)
b_out = np.zeros(n_out)
'''
W_in  = np.random.normal(0, np.sqrt(1/(n_in)), (n_hidden, n_in))
W_rec = np.random.normal(0, np.sqrt(1/n_hidden), (n_hidden, n_hidden))
W_out = np.random.normal(0, np.sqrt(1/(n_hidden)), (n_out, n_hidden))

b_rec = np.zeros(n_hidden)
b_out = np.zeros(n_out)
'''

#alpha: to what extent new gradient will be used.
alpha = 1
#sigma (float): Specifies standard deviation of white noise.
sigma = 0

rnn = GRU(Wz_in,Wr_in,Wh_in, Wz_rec,Wr_rec,Wh_rec, W_out, bz_rec, br_rec, bh_rec, b_out,
          activation=tanh,
          alpha=alpha,
          output=identity,
          loss=mean_squared_error)
'''
rnn = RNN(W_in, W_rec, W_out, b_rec, b_out,
          activation=relu,
          alpha=alpha,
          output=identity,
          loss=mean_squared_error)
'''

#activation: tanh
#relu less stable, less lr for stablization
learn_alg = Efficient_BPTT(rnn, T_truncation=6)
#learn_alg = DNI(rnn, sg_optimizer)
#learn_alg = KF_RTRL(rnn, L2_reg=0.0001, L1_reg=0.0001)
#learn_alg = RFLO(rnn, alpha=alpha, L2_reg=0.0001, L1_reg=0.0001)

#mu= to what extent old gradient will be used.
#add clip_norm = True for gru
optimizer = SGD_Momentum(lr=0.001, mu=0.6)
#optimizer = Stochastic_Gradient_Descent(lr=params['sg_lr'])


monitors = []

sim = Simulation(rnn)
sim.run(data, learn_alg=learn_alg, optimizer=optimizer,
        sigma=sigma,
        monitors=monitors,
        verbose=True,
        report_accuracy=False,
        report_loss=True,
        checkpoint_interval=checkpoint_interval)

test_sim = Simulation(rnn)
test_sim.run(data, mode='test', monitors=['rnn.y_hat','rnn.loss_'], verbose=False)
processed_data = test_sim.mons['rnn.loss_']
#test_sim.mos: a dictionary of monitors
#test_sim.mons['rnn.loss_']: a numpy ndarray
print(np.mean(processed_data))
acc = classification_accuracy(data, test_sim.mons['rnn.y_hat'])
print(acc)