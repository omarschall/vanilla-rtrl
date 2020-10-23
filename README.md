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
**test_{}.py**: Contains unit tests for some of the modules, most crucially learning_algorithms.py, network.py, getn_data.py, and utils.py.  
**utils.py**: Random useful functions.  
