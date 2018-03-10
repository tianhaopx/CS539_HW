from problem1 import SoftmaxRegression as sr
import torch as th
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import SGD

# -------------------------------------------------------------------------
'''
    Problem 2: Convolutional Neural Network 
    In this problem, you will implement a convolutional neural network with a convolution layer and a max pooling layer.
    The goal of this problem is to learn the details of convolutional neural network. 
    You could test the correctness of your code by typing `nosetests -v test2.py` in the terminal.
    Note: please do NOT use th.nn.functional.conv2d or th.nn.Conv2D, implement your own version of 2d convolution using only basic tensor operations.
'''


# --------------------------
def conv2d(x, W, b):
    '''
        Compute the 2D convolution with one filter on one image, (assuming stride=1).
        Input:
            x:  one training instance, a float torch Tensor of shape l by h by w. 
                h and w are height and width of an image. l is the number channels of the image, for example RGB color images have 3 channels.
            W: the weight matrix of a convolutional filter, a torch Variable of shape l by s by s. 
            b: the bias vector of the convolutional filter, a torch scalar Variable. 
        Output:
            z: the linear logit tensor after convolution, a float torch Variable of shape (h-s+1) by (w-s+1)
        Note: please do NOT use th.nn.functional.conv2d, implement your own version of 2d convolution,using basic tensor operation, such as dot().
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    l, h, w = x.size()
    s = W.size()[1]
    z = Variable(th.zeros(h - s + 1, w - s + 1))
    for i in xrange(l):
        # c = x[i]
        for j in xrange(h - s + 1):
            for k in xrange(w - s + 1):
                z[j, k] = z[j, k] + th.dot(x[i, j:j + s, k:k + s], W[i])
    z = z + b

    #########################################
    return z


# --------------------------
def Conv2D(x, W, b):
    '''
        Compute the 2D convolution with multiple filters on a batch of images, (assuming stride=1).
        Input:
            x:  a batch of training instances, a float torch Tensor of shape (n by l by h by w). n is the number instances in a batch.
                h and w are height and width of an image. l is the number channels of the image, for example RGB color images have 3 channels.
            W: the weight matrix of a convolutional filter, a torch Variable of shape (n_filters by l by s by s). 
            b: the bias vector of the convolutional filter, a torch vector Variable of length n_filters. 
        Output:
            z: the linear logit tensor after convolution, a float torch Variable of shape (n by n_filters by (h-s+1) by (w-s+1) )
        Note: please do NOT use th.nn.functional.conv2d, implement your own version of 2d convolution,using basic tensor operation, such as dot().
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    n, l, h, w = x.size()
    n_filters = W.size()[0]
    s = W.size()[2]
    z = Variable(th.zeros(n, n_filters, h - s + 1, w - s + 1))
    for i in xrange(n):
        for j in xrange(n_filters):
            z[i, j] = conv2d(x[i], W[j], b[j])

    #########################################
    return z


# --------------------------
def ReLU(z):
    '''
        Compute ReLU activation. 
        Input:
            z: the linear logit tensor after convolution, a float torch Variable of shape (n by n_filters by h by w )
                h and w are the height and width of the image after convolution. 
        Output:
            a: the nonlinear activation tensor, a float torch Variable of shape (n by n_filters by h by w )
        Note: please do NOT use th.nn.functional.relu, implement your own version using only basic tensor operations. 
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    n, n_filters, h, w = z.size()
    a = Variable(th.zeros(n, n_filters, h, w))
    for i in xrange(n):
        for j in xrange(n_filters):
            for k in xrange(h):
                for l in xrange(w):
                    if float(z[i, j, k, l]) > 0:
                        a[i, j, k, l] = z[i, j, k, l]
                    else:
                        a[i, j, k, l] = 0

    #########################################
    return a


# --------------------------
def avgpooling(a):
    '''
        Compute the 2D average pooling (assuming shape of the pooling window is 2 by 2).
        Input:
            a:  the feature map of one instance, a float torch Tensor of shape (n by n_filter by h by w). n is the batch size, n_filter is the number of filters in Conv2D.
                h and w are height and width after ReLU. 
        Output:
            p: the tensor after pooling, a float torch Variable of shape n by n_filter by floor(h/2) by floor(w/2).
        Note: please do NOT use torch.nn.AvgPool2d or torch.nn.functional.avg_pool2d or avg_pool1d, implement your own version using only basic tensor operations.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    n, n_filters, h, w = a.size()
    p = Variable(th.zeros(n, n_filters, h // 2, w // 2))
    for i in xrange(n):
        for j in xrange(n_filters):
            for k in xrange(h // 2):
                for l in xrange(w // 2):
                    p[i, j, k, l] = th.mean(a[i, j, k * 2:k * 2 + 2, l * 2:l * 2 + 2])

    #########################################
    return p


# --------------------------
def maxpooling(a):
    '''
        Compute the 2D max pooling (assuming shape of the pooling window is 2 by 2).
        Input:
            a:  the feature map of one instance, a float torch Tensor of shape (n by n_filter by h by w). n is the batch size, n_filter is the number of filters in Conv2D.
                h and w are height and width after ReLU. 
        Output:
            p: the tensor after max pooling, a float torch Variable of shape n by n_filter by floor(h/2) by floor(w/2).
        Note: please do NOT use torch.nn.MaxPool2d or torch.nn.functional.max_pool2d or max_pool1d, implement your own version using only basic tensor operations.
        Note: if there are mulitple max values, select the one with the smallest index.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    n, n_filters, h, w = a.size()
    p = Variable(th.zeros(n, n_filters, h // 2, w // 2))
    for i in xrange(n):
        for j in xrange(n_filters):
            for k in xrange(h // 2):
                for l in xrange(w // 2):
                    tmp = a[i, j, k * 2:k * 2 + 2, l * 2:l * 2 + 2]
                    max = tmp[0, 0]
                    for x in xrange(2):
                        for y in xrange(2):
                            if tmp[x, y].data[0] > max.data[0]:
                                max = tmp[x, y]
                    # p[i, j, k, l] = th.max(a[i, j, k * 2:k * 2 + 2, l * 2:l * 2 + 2])
                    p[i, j, k, l] = max
                    #########################################
    return p


# --------------------------
def num_flat_features(h=28, w=28, s=3, n_filters=10):
    ''' Compute the number of flat features after convolution and pooling. Here we assume the stride of convolution is 1, the size of pooling kernel is 2 by 2, no padding. 
        Inputs:
            h: the hight of the input image, an integer scalar
            w: the width of the input image, an integer scalar
            s: the size of convolutional filter, an integer scalar. For example, a 3 by 3 filter has a size 3.
            n_filters: the number of convolutional filters, an integer scalar
        Outputs:
            p: the number of features we will have on each instance after convolution, pooling, and flattening, an integer scalar.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    h_ = h - s + 1
    w_ = w - s + 1
    p = (h_ // 2) * (w_ // 2) * n_filters

    #########################################
    return p


# -------------------------------------------------------
class CNN(sr):
    '''CNN is a convolutional neural network with a convolution layer (with ReLU activation), a max pooling layer and a fully connected layer.
       In the convolutional layer, we will use ReLU as the activation function. 
       After the convolutional layer, we apply a 2 by 2 max pooling layer, before feeding into the fully connected layer.
    '''

    # ----------------------------------------------
    def __init__(self, l=1, h=28, w=28, s=5, n_filters=5, c=10):
        ''' Initialize the model. Create parameters of convolutional layer and fully connected layer. 
            Inputs:
                l: the number of channels in the input image, an integer scalar
                h: the hight of the input image, an integer scalar
                w: the width of the input image, an integer scalar
                s: the size of convolutional filter, an integer scalar. For example, a 3 by 3 filter has a size 3.
                n_filters: the number of convolutional filters, an integer scalar
                c: the number of output classes, an integer scalar
            Outputs:
                self.conv_W: the weight matrix of the convolutional filters, a torch Variable of shape n_filters by l by s by s, initialized as all-zeros. 
                self.conv_b: the bias vector of the convolutional filters, a torch vector Variable of length n_filters, initialized as all-ones, to avoid vanishing gradient.
                self.W: the weight matrix parameter in fully connected layer, a torch Variable of shape (p, c), initialized as all-zeros. 
                        Hint: CNN is a subclass of SoftmaxRegression, which already has a W parameter. p is the number of flat features after pooling layer.
                self.b: the bias vector parameter, a torch Variable of shape (c), initialized as all-zeros
                self.loss_fn: the loss function object for softmax regression. 
            Note: In this problem, the parameters are initialized as either all-zeros or all-ones for testing purpose only. In real-world cases, we usually initialize them with random values.
        '''
        #########################################
        ## INSERT YOUR CODE HERE

        # compute the number of flat features
        p = num_flat_features(h, w, s, n_filters)

        # initialize fully connected layer
        self.conv_W = Variable(th.zeros(n_filters, l, s, s), requires_grad=True)
        self.conv_b = Variable(th.ones(n_filters), requires_grad=True)
        super(CNN, self).__init__(p, c)

        # the kernel matrix of convolutional layer 

        #########################################

    # ----------------------------------------------
    def forward(self, x):
        '''
           Given a batch of training instances, compute the linear logits of the outputs. 
            Input:
                x:  a batch of training instance, a float torch Tensor of shape n by l by h by w. Here n is the batch size. l is the number of channels. h and w are height and width of an image. 
            Output:
                z: the logit values of the batch of training instances after the fully connected layer, a float matrix of shape n by c. Here c is the number of classes
        '''
        #########################################
        ## INSERT YOUR CODE HERE

        # convolutional layer
        z = Conv2D(x, self.conv_W, self.conv_b)

        # ReLU activation
        a = ReLU(z)

        # maxpooling layer
        pooling = maxpooling(a)

        # flatten
        flatten = pooling.view(pooling.size()[0], -1)

        # fully connected layer
        z = super(CNN, self).forward(flatten)

        #########################################
        return z

    # ----------------------------------------------
    def train(self, loader, n_steps=10, alpha=0.01):
        """train the model 
              Input:
                loader: dataset loader, which loads one batch of dataset at a time.
                n_steps: the number of batches of data to train, an integer scalar
                alpha: the learning rate for SGD(stochastic gradient descent), a float scalar
        """
        # create a SGD optimizer
        optimizer = SGD([self.conv_W, self.conv_b, self.W, self.b], lr=alpha)
        count = 0
        while True:
            # use loader to load one batch of training data
            for x, y in loader:
                # convert data tensors into Variables
                x = Variable(x)
                y = Variable(y)
                #########################################
                ## INSERT YOUR CODE HERE

                # forward pass
                z = self.forward(x)

                # compute loss
                L = super(CNN, self).compute_L(z, y)

                # backward pass: compute gradients
                super(CNN, self).backward(L)

                # update model parameters
                optimizer.step()

                # reset the gradients
                optimizer.zero_grad()

                #########################################
                count += 1
                if count >= n_steps:
                    return
