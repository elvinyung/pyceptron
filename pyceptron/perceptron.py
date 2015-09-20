
import random

class VectorLengthMismatch(Exception):
    pass

class SingleLayerPerceptron:
    def __init__(self, dimensions=2, learning_rate=0.1, threshold=0.5):
        self.dimensions = dimensions
        self.weights = tuple(random.random() for d in range(self.dimensions))
        self.bias = 1.0
        self.threshold = threshold

        self.learning_rate = learning_rate

        self.error = 0.0
        self.error_threshold = 0.0

    def activate(self, value):
        '''
        Basic Heaviside activation function.
        '''
        return 1 if value >= self.threshold else 0

    def weight_dot_product(self, vector):
        '''
        Dot product vector with weights (must be same dimensions).
        '''
        if len(vector) != len(self.weights):
            raise VectorLengthMismatch
        return sum(w * x for (w, x) in zip(self.weights, vector)) + self.bias

    def get_output(self, vector):
        '''
        Layer output function.
        Essentially the dot product of the weights and the input vector, passed
        through the Heaviside activation function.
        '''
        return self.activate(self.weight_dot_product(vector))

    def learn_from(self, input_vector, expected_output):
        '''
        Learn from a single input-output pair, adjusting the perceptron's
        weights and biases.
        Returns the sample error, i.e. the discrepancy between the expected
        output and the perceptron output.
        '''
        output = self.get_output(input_vector)
        sample_error = expected_output - output

        if sample_error:
            # update weights and bias
            adjustment = sample_error * self.learning_rate
            self.weights = tuple(w + (adjustment * x)
                for (w, x) in zip(self.weights, vector))
            self.bias += adjustment

        return sample_error

    def iterate(self, dataset):
        '''
        Run one iteration of training.
        Returns the iteration error.

        The dataset has the format of an array containing dictionaries with
        keys 'input' (the input vector), and 'output' (the output value).
        '''

        iter_error = 0.0
        for example in dataset:
            # learn from each input-output pair.
            sample_error = self.learn_from(example['input'], example['output'])

            # record iteration error
            iter_error += abs(sample_error)

        # final iteration error is the average error in the iteration.
        iter_error /= len(dataset)

        return iter_error

    def train(self, dataset, max_iterations=0):
        '''
        Train the perceptron on a dataset.
        i.e. continuously iterate until the error is below threshold,
        or the max_iterations has been reached.
        '''
        num_iterations = 0
        iter_error = self.error_threshold + 1.0 # hack
        while ((max_iterations <= 0 or num_iterations < max_iterations) and
            (iter_error > self.error_threshold)):
            iter_error = self.iterate(dataset)
            num_iterations += 1
