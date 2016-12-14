#!/usr/bin/env python3
# By Jon Dehdari, 2016

"""
Perceptron demo.

Created by Jon Dehdari, revised and commented by Dominik Macháček.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

labels = [0, 1]
#num_samples = 30
num_samples = 200
np.random.seed(seed=3)
learning_rate = 1.0


## the percentage of test_set from the whole data
# test_percentage = 10
# test_size = num_samples // test_percentage
# train_size = num_samples - test_size

##### Comment: the name of `test_percentage` variable is confusing, it doesn't fit reality, in fact in main there 
##### are `num_samples` of blue points and `num_samples` of red points,
##### so 2*`num_samples` of points in total. Therefore I change it this way:

total_samples = 2*num_samples
## the percentage of test_set size from the whole data
test_percentage = 20
test_size = total_samples * test_percentage // 100
train_size = total_samples - test_size


class Perceptron(list):
    """
    Perceptron is a binary classifier.
    During learning phase it finds one hyperplane which separates whole feature space to two classes, and then 
    on predicting it returns the class by the side of hyperplane of particular hyper-point.

    Example:
    Prepare data for Perceptron:
    >>> fig = plt.figure()
    >>> subplot = fig.add_subplot(1,1,1, xlim=(-5,5), ylim=(-5,5))
    >>> train_set = [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 1)]
    >>> test_set  = [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 1)]
    
    Create Perceptron for classification of data with 2 features
    >>> p = Perceptron(2)
    
    Train it on `train_set`, count and print accuracy on `test_set` every `status` learning iteration,
    plot it to `subplot` and `fig`:
    >>> p.train(train_set, test_set, status=10, subplot=subplot, fig=fig)
    
    Predicted class for vector (30,20):
    >>> p.predict((30,20))
    
    To get parameters of a hyperplane:
    >>> p.bias, p.params
    it's formula (n is dimension):
     p.params[0]*x_0 + p.params[1]*x_1 + ... + p.params[n-1]*x_(n-1) + p.bias = 0
    """

    def __init__(self, dimension, learning_rate=learning_rate):
        """:arg dimension: dimension of data
        :arg learning_rate: initial learning rate
        
		"""
        self.dimension = dimension        
        # np.array of two random numbers from normal (Gaussian) distribution with mean 0 and standard deviation 1
        self.params = np.random.normal(size=dimension)
        #print(type(self.params))
        # TODO: if params is set to [0,0] (which can happen from np.random.normal) like this
        # self.params = np.array([0,0])
        # and self.bias = 0, which can happen as well, then 
        # perceptron.py:123: RuntimeWarning: divide by zero encountered in true_divide
        # subplot.plot([0, -self.bias / self.params[1]], [-self.bias / self.params[0], 0], '-', color='lightgreen', linewidth=1)
        
#        self.params = np.array([0.00001,0.00001])
        # random number from normal (Gaussian) distribution with mean 0 and standard deviation 1
        self.bias   = np.random.normal()
        self.bias = 1
        self.learning_rate = learning_rate


    def __repr__(self):
        return str(np.concatenate((self.params, [self.bias])))


    def error(self, guess, correct):
        """Assume guess is a predicted class by perceptron, correct is the correct class.
        If class labels are 0 and 1, this function returns 0, if the guess was correct, 
        or -1, or 1, the direction of error (the guess was too high or too low)."""
        return correct - guess


    def activate(self, val):
        """This function assigns a label according to the output of the perceptron.
        :arg val: perceptron's output
        :returns 0 or 1 by the side of vector and hyperplane.
        """
        if val >= 0:
            return 1
        else:
            return 0


    def predict(self, x):
        """:args x: a feature vector
        Predicts a class to x according to its position in feature space and the hyper-plane from training.
        """
        return self.activate(np.dot(x, self.params) + self.bias)  # it's a test whether a vector lies under a hyper-plane, 
        # above it or on it


    def predict_set(self, test_set):
        """:return accuracy on test_set"""
        errors = 0
        for x, y in test_set:
            out = self.predict(x)
            if out != y:
                errors +=1  
        return 1 - errors / len(test_set)


    def decision_boundary(self):
        """ Returns two points, along which the decision boundary for a binary classifier lies. """
        return ((0, -self.bias / self.params[1]), (-self.bias / self.params[0], 0))


    def train(self, train_set, dev_set, status=100, epochs=10, subplot=None, fig=None):
        """:arg train_set: perceptron's training test 
         :arg dev_set: dev test set, used only for informative testing during learning
         
         train_set and dev_set must be lists of format [((n-tuple with feature values), label), ...], where n is )
         the dimension for which perceptron is created, label is 0 or 1
         
         :arg status: only every `status` iteration, status will be printed 
         :arg epochs: number of training epochs (iterations trough training data). It's hardcoded by this argument, 
         but in future versions it could be set dynamically by the learning performance (e.g. stop if the accuracy stagnates
         or oscillates).
         
         :arg subplot: for plotting data during training
         :arg fig: for plotting data
         """
        
        if not all(map(lambda dim_lab: dim_lab[0]==self.dimension and dim_lab[1] in [0,1], 
                   ((len(tup), label) for tup,label in train_set + dev_set))):
            raise ValueError("invalid training or dev_testing data")
        
        
        print("Starting dev set accuracy: %g; Line at %s; Params=%s" % (self.predict_set(dev_set), str(self.decision_boundary()), str(self)), file=sys.stderr)
        subplot.plot([0, -self.bias/self.params[1]], [-self.bias/self.params[0], 0], '-', color='lightgreen', linewidth=1)
        fig.canvas.draw()

        iterations = 0
        # number of iterations over the whole training data
        for epoch in range(epochs):
            np.random.shuffle(train_set)  # we shuffle the training set
            for x, y in train_set:
                iterations += 1
                # guess label
                out = self.predict(x)
                # error direction, if the guess was incorrect we slightly move the hyperplane towards x or back from 
                # it
                error = self.error(out, y)
                if error != 0:
                    #print("out=", out, "; y=", y, "; error=", error, file=sys.stderr)
                    
                    # learning rate is the rate how much we move (modify) the hyperplane
                    self.bias += self.learning_rate * error # ...
                    for i in range(len(x)):
                        self.params[i] += self.learning_rate * error * x[i]  # ...
                # print status
                if iterations % status == 0:
                    print("Dev set accuracy: %g; Line at %s; Params=%s" % (self.predict_set(dev_set), str(self.decision_boundary()), str(self)), file=sys.stderr)
                    subplot.plot([0, -self.bias/self.params[1]], [-self.bias/self.params[0], 0], '-', color='lightgreen', linewidth=1)
                    fig.canvas.draw()
            # in every epoch learning rate is diminished so it does smaller and smaller steps in every iteration because the
            # accuracy is supposed to be better. If we wouldn't do it, then the he hyperplane could oscillate between 
            # same values and skip the best one,
            # or one incorrect example in last epoch could move a lot and destroy good-performing hyperplane.
            self.learning_rate *= 0.9
        subplot.plot([0, -self.bias/self.params[1]], [-self.bias/self.params[0], 0], 'g-x', linewidth=5)
        fig.canvas.draw()


def main():
    import doctest
    doctest.testmod()  # check the code snippets in docstrings 

    # we have a 2D plot:
    
    # we randomly generate `num_samples` of blue points...
    # (their distribution is Gaussian, mean=0, standard deviation is 1 by x axis and 0.5 by y axis)
    # so they're located mostly within an imaginary ellipse with center (0,0), horizontal axis has length 1 and vertical 0.5
    x_blue = np.random.normal(loc=0, size=num_samples)
    y_blue = np.random.normal(loc=0, scale=0.5, size=num_samples)
    
    # and `num_samples` of blue points, their scattered mostly within 
    # an imaginary ellipse with center (2,2), horizontal axis has length 1 and vertical 0.5
    x_red  = np.random.normal(loc=2, size=num_samples)
    y_red  = np.random.normal(loc=2, scale=0.5, size=num_samples)
    
    # one element in data will be ((x_position, y_position), color)
    data =  list(zip(zip(x_blue, y_blue), [labels[0]] * num_samples))
    data += zip(zip(x_red,  y_red),  [labels[1]] * num_samples)
    np.random.shuffle(data) 
    # data is now an array consisting of all blue and red points, it's shuffled
    
    # we split data to train and test_set
    train_set = data[:train_size]
    test_set  = data[train_size:]
    
    # Matplotlib craziness to be able to update a plot
    plt.ion()
    fig = plt.figure()
    subplot = fig.add_subplot(1,1,1, xlim=(-5,5), ylim=(-5,5))
    subplot.grid()
    subplot.plot(x_blue, y_blue, linestyle='None', marker='o', color='blue')
    subplot.plot(x_red,  y_red,  linestyle='None', marker='o', color='red')
    
    # we create a perceptron
    p = Perceptron(2)
    # train it
    p.train(train_set, test_set, subplot=subplot, fig=fig)


if __name__ == '__main__':
    main()
