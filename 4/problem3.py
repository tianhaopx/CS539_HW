from problem1 import SoftmaxRegression as sr
import torch as th
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import SGD

# -------------------------------------------------------------------------
'''
    Problem 3: Recurrent Neural Network 
    In this problem, you will implement the recurrent neural network for sequence classification problems.
    We will use cross entropy as the loss function and stochastic gradient descent to train the model parameters.
    You could test the correctness of your code by typing `nosetests test3.py` in the terminal.
    Note: please do NOT use torch.nn.RNN, implement your own version of RNN using only basic tensor operations.
'''


# --------------------------
def tanh(z):
    """ Compute the hyperbolic tangent of the elements
        math: f(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z))
        Input:
            z:  a batch of training instances, a float torch Variable of shape (n by p). n is the number instances in a batch. p is the number of features.
        Output:
            a: the nonlinear activation tensor, a float torch Variable of shape (n by n_filters by h by w )
        Note: please do NOT use th.nn.Tanh, th.tanh, th.nn.functional.tanh, implement your own version using basic tensor operations, such as exp(), div(), etc.
    """
    #########################################
    ## INSERT YOUR CODE HERE
    # a = th.div(th.exp(z)-th.exp(-z),th.exp(z)+th.exp(-z))
    raw_size = z.size()
    z = z.view(-1)
    a = Variable(th.zeros(z.size()[0]))
    for i in xrange(z.size()[0]):
        # for j in xrange(p):
        if z[i].data[0] > 600:
            a[i] = 1
        elif z[i].data[0] < -600:
            a[i] = -1
        else:
            a[i] = th.div(th.exp(z[i]) - th.exp(-z[i]), th.exp(z[i]) + th.exp(-z[i]))
    a = a.view(*raw_size)

    #########################################
    return a


# -------------------------------------------------------
class RNN(sr):
    '''RNN is a recurrent neural network with hyperbolic tangent function as the activation function.
       After the recurrent layer, we apply a fully connected layer from hidden neurons to produce the output.
    '''

    # ----------------------------------------------
    def __init__(self, p, h=10, c=10):
        ''' Initialize the model. Create parameters of recurrent neural network. 
            Inputs:
                p: the number of input features, an integer scalar
                h: the number of memory (hidden) neurons, an integer scalar
                c: the number of output classes, an integer scalar
            Outputs:
                self.U: the weight matrix connecting the input features to the hidden units, a torch Variable of shape p by h, initialized as all-zeros. 
                self.V: the weight matrix connecting the hidden units to hidden units, a torch vector Variable of shape h by h, initialized as all-zeros.
                self.b_h: the bias vector of the hidden units, a torch vector Variable of length h, initialized as all-ones, to avoid vanishing gradient.
                self.W: the weight matrix parameter in fully connected layer from hidden units to the output prediction, a torch Variable of shape (h, c), initialized as all-zeros. 
                        Hint: RNN is a subclass of SoftmaxRegression, which already has a W parameter and b. 
                self.b: the bias vector parameter of the outputs, a torch Variable of shape (c), initialized as all-zeros
                self.loss_fn: the loss function object for softmax regression. 
            You could solve this problem using 4 lines of code.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        self.U = Variable(th.zeros(p, h), requires_grad=True)
        self.V = Variable(th.zeros(h, h), requires_grad=True)
        self.b_h = Variable(th.ones(h), requires_grad=True)
        super(RNN, self).__init__(h, c)

        #########################################

    # ----------------------------------------------
    def forward(self, x, H):
        '''
           Given a batch of training instances (with one time step), compute the linear logits z in the outputs.
            Input:
                x:  a batch of training instance, a float torch Tensor of shape n by p. Here n is the batch size. p is the number of features. 
                H:  the hidden state of the RNN model, a float torch Tensor of shape  n by h. Here h is the number of hidden units. 
            Output:
                z: the logit values of the batch of training instances after the output layer, a float matrix of shape n by c. Here c is the number of classes
                H_new: the new hidden state of the RNN model, a float torch Tensor of shape n by h.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        H_new = tanh(th.mm(x, self.U) + th.mm(H, self.V) + self.b_h)
        z = th.mm(H_new, self.W) + self.b
        #########################################
        return z, H_new

    # ----------------------------------------------
    def train(self, loader, n_steps=10, alpha=0.01):
        """train the model 
              Input:
                loader: dataset loader, which loads one batch of dataset at a time.
                        x: a batch of training instance, a float torch Tensor of shape n by t by p. Here n is the batch size. p is the number of features. t is the number of time steps.
                        y: a batch of training labels, a torch LongTensor of shape n by t. 
                n_steps: the number of batches of data to train, an integer scalar. Note: the n_steps is the number of training steps, not the number of time steps (t).
                alpha: the learning rate for SGD(stochastic gradient descent), a float scalar
                Note: the loss of a sequence is computed as the sum of the losses in all time steps of the sequence.
        """
        # create a SGD optimizer
        optimizer = SGD([self.U, self.V, self.b_h, self.W, self.b], lr=alpha)
        count = 0
        while True:
            # use loader to load one batch of training data
            for x, y in loader:
                # convert data tensors into Variables
                x = Variable(x)
                y = Variable(y)
                n, t, p = x.size()
                h, _ = self.V.size()
                # initialize hidden state as all zeros
                H = Variable(th.zeros(n, h))
                #########################################
                ## INSERT YOUR CODE HERE
                L = 0
                # go through each time step
                for i in xrange(t):
                    # forward pass
                    z, H = self.forward(x[:, i, :], H)
                    # compute loss
                    L += super(RNN, self).compute_L(z, y[:, i])

                # backward pass: compute gradients
                super(RNN, self).backward(L)

                # update model parameters
                optimizer.step()

                # reset the gradients to zero
                optimizer.zero_grad()

                #########################################
                count += 1
                if count >= n_steps:
                    return
