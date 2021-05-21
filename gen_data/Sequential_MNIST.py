import numpy as np
import pickle
from .Task import Task

class Sequential_MNIST(Task):
    """Class for the Sequential MNIST task, where chunks of a given MNIST
    image are fed into the network at each time step. The number of pixels
    per time step parameterizes the difficulty of the task.

    Note: uses the numpy mnist pickle file mnist.pkl as created from the
    GitHub repo https://github.com/hsjeong5/MNIST-for-Numpy/."""

    def __init__(self, pixels_per_time_step):
        """Inits an instance of Sequential_MNIST by specifying the number
        of pixels per time step."""

        if (784 % pixels_per_time_step) != 0:
            raise ValueError('The number of pixels per time step must ' +
                             'divide the total pixels per image')

        super().__init__(pixels_per_time_step, 10)

        self.pixels_per_time_step = pixels_per_time_step
        self.inputs_per_image = 784 // self.pixels_per_time_step

        with open('files/data/mnist.pkl', 'rb') as f:
            mnist = pickle.load(f)

        self.mnist_images = np.concatenate([mnist['training_images'],
                                            mnist['test_images']], axis=0)
        self.mnist_labels = np.concatenate([mnist['training_labels'],
                                            mnist['test_labels']])

    def gen_dataset(self, N):
        """Generates a test set by taking the concatenated training/test
        images and labels, randomly sampling from each, and then reshaping
        them into the form specified by pixels_per_time_step."""

        N_images = N // self.inputs_per_image
        image_indices = np.random.choice(list(range(70000)), N_images,
                                         replace=False)
        mnist_images = self.mnist_images[image_indices]
        mnist_labels = self.mnist_labels[image_indices]

        mnist_images = mnist_images.reshape((-1, self.pixels_per_time_step))
        one_hot_labels = np.squeeze(np.eye(10)[mnist_labels])

        X = mnist_images
        Y = np.tile(one_hot_labels, self.inputs_per_image).reshape((-1, 10))

        return X, Y
