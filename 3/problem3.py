import math
import numpy as np
from problem2 import DT
from collections import Counter

# -------------------------------------------------------------------------
'''
    Problem 3: Bagging: Boostrap Aggregation of decision trees (on continous attributes)
    You could test the correctness of your code by typing `nosetests -v test3.py` in the terminal.
'''


# -----------------------------------------------
class Bag(DT):
    '''
        Bagging ensemble of Decision Tree (with contineous attributes)
        Hint: Bagging is a subclass of DT class in problem2. So you can reuse and overwrite the code in problem 2.
    '''

    # --------------------------
    @staticmethod
    def bootstrap(X, Y):
        '''
            Create a boostrap sample of the dataset. 
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
            Output:
                X: the bootstrap sample of the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the bootstrap sample of the class labels, a numpy array of length n. Each element can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        p, n = X.shape
        newX = []
        newY = []
        X = np.asarray(X)
        # Y = np.asarray(Y)
        for _ in xrange(n):
            i = np.random.randint(0, n)
            newX.append(X[:, i])
            newY.append(Y[i])
        X = np.asarray(newX).T
        Y = np.asarray(newY)

        #########################################
        return X, Y

        # --------------------------

    def train(self, X, Y, n_tree=11):
        '''
            Given a training set, train a bagging ensemble of decision trees. 
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the training set, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
                n_tree: the number of trees in the ensemble
            Output:
                T: a list of the root of each tree, a list of length n_tree.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        T = []
        for _ in xrange(n_tree):
            newX, newY = self.bootstrap(X, Y)
            T.append(DT.train(self, X, Y))
        #########################################
        return T

        # --------------------------

    @staticmethod
    def inference(T, x):
        '''
            Given a bagging ensemble of decision trees and one data instance, infer the label of the instance. 
            Input:
                T: a list of decision trees.
                x: the attribute vector, a numpy vectr of shape p.
                   Each attribute value can be int/float
            Output:
                y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        res = [DT.inference(t, x) for t in T]
        y = Counter(res).most_common(1)[0][0]
        #########################################
        return y

    # --------------------------
    @staticmethod
    def predict(T, X):
        '''
            Given a decision tree and a dataset, predict the labels on the dataset. 
            Input:
                T: a list of decision trees.
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float.
                   Here n is the number data instances in the dataset, p is the number of attributes.
            Output:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        Y = []
        for i in range(X.shape[1]):
            Y.append(Bag.inference(T, X[:, i]))
        Y = np.asarray(Y)
        #########################################
        return Y

    # --------------------------
    @staticmethod
    def load_dataset(filename='data3.csv'):
        '''
            Load dataset 3 from the CSV file:data3.csv. 
            The first row of the file is the header (including the names of the attributes)
            In the remaining rows, each row represents one data instance.
            The first column of the file is the label to be predicted.
            In remaining columns, each column represents an attribute.
            Input:
                filename: the filename of the dataset, a string.
            Output:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the dataset, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        data = np.genfromtxt(filename, dtype='object', delimiter=',', skip_header=1).T
        Y = data[0].astype(int)
        X = data[1:, :].astype(float)
        #########################################
        return X, Y
