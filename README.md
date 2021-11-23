# Updates about GRU
Changes are made to:
1. models: implements GRU.py, and add a new property for RNN.py to identify between them
2. Simulation.py: changes the updating process in self.train_step() since there are more parameters in GRU than in RNN.
3. Learning Algotirithms: for now only Efficient_BPTT and RFLO are implemented, others remaining the same.
Learning_Algorithm.py, RFLO.py, Efficient_BPTT: update GRU parts, and work for GRU and RNN by identifying the new property self.type in GRU.py and RNN.py.
4. Optimizers: no changes.
# Key problems
1. recurrent update: GRU has its own update gate, so I wonder if we still need self.alpha, but I keep them both for now.
2. propagation: From my test result, I think updates for get_a_jacobian() in GRU.py and Learning algorithms may have some mistakes. You may examine these parts if you have time.
3. overflow problem: I firstly implemented Efficient_BPTT, and it caused considerable overflow problems when lr = 0.01. I use clip_norm to fix the problem.
# Test cases
1. model: RNN, learn_alg = RFLO(rnn, alpha=1, L2_reg=0.0001, L1_reg=0.0001),optimizer = SGD_Momentum(lr=0.01, mu=0.6)
Works as usual. Proves the integration works well.
2. model: GRU, learn_alg = Efficient_BPTT(rnn, T_truncation=6),optimizer = SGD_Momentum(lr=0.001, mu=0.6, clip_norm=True)
loss: 0.205. Used clip_norm.
3. model: GRU, learn_alg = RFLO(rnn, alpha=1, L2_reg=0.0001, L1_reg=0.0001),optimizer = SGD_Momentum(lr=0.01, mu=0.6)
loss: 0.036.

# vanilla-rtrl
Real-time recurrent learning and approximations in traditional setting

Run main.py to train a network and observe its behvaior after training. In main, the following steps are taken:

1. Create a Task object from gen_data.Task.
2. Use the Task object to generate training (and testing) inputs and labels, as a dictionary called data.
3. Create an RNN object using some initial parameter values (numpy arrays) and activation/output/loss functions (instances of functions.Function class).
4. Create a Learning_Algorithm object from learning_algorithms.Learning_Algorithm.
5. Create an Optimizer object from optimizers.Optimizer.
6. Create a Simulation object from simulation.Simulation referencing the RNN object.
7. Create a list of strings called monitors, which are essentially addresses to some attribute or child attirube of the Simulation object whose values should be recorded at each time step.
8. Call the run() method on the simulation object, taking as argument the data dictionary, the the Learning_Algorithm and Optimizer instances, and other arguments like monitors. Loops through the data sequentially (batch size of 1), running the RNN forwards, calculating errors, and updating the RNN if in 'train' mode.
9. Run new simulations in 'test' mode to evaluate network behvaior and plot relevant results.

Other modules:  
**functions.py**: Contains Function class (for compactly storing a function and its derivative) and several key instances, such as tanh and softmax cross entropy.  
**gen_data.py**: Contains Tas class and several subclasses for specific types of tasks, such as Add and Mimic.  
**learning_algorithms.py**: Contains Learning_Algorithm parent class and several subclasses for specific types of learning algorithms, such as RTRL, BPTT, UORO, etc.  
**network.py**: Contains RNN class, which defines a leaky vanilla RNN. (In principle, there could be a general RNN class with subclasses for specific RNN architectures, with the Learning_Algorithm subclasses written to be as architecture-agnostic as possible. However, we currently have hardcoded in leaky vanilla RNNs.)  
**optimizers.py**: Contains Optimizer class, which provides functions that take gradients from the Learning_Algorithm instance and uses them to update the RNN parameters. Only specific subclass is Stochastic_Gradient_Descent, but one could define other optimizers such as Adam or RMSProp.  
**simulation.py**: Contains Simulation class, which takes in all other types of objects and simulates an RNN either in 'train' or 'test' mode.  
**submit_jobs.py**: Contains functions for quickly running grid parameter searches on the NYU high-performance computing machines. Likely to not be useful for anyone other than me.   
**test_{}.py**: Contains unit tests for some of the modules, most crucially learning_algorithms.py, network.py, getn_data.py, and utils.py.  
**utils.py**: Random useful functions.  
