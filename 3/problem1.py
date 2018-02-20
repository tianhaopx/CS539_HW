import math
import numpy as np
from collections import Counter

# -------------------------------------------------------------------------
'''
    Problem 1: Decision Tree (with Descrete Attributes)
    In this problem, you will implement the decision tree method for classification problems.
    You could test the correctness of your code by typing `nosetests -v test1.py` in the terminal.
'''


# -----------------------------------------------
class Node:
    '''
        Decision Tree Node (with discrete attributes)
        Inputs: 
            X: the data instances in the node, a numpy matrix of shape p by n.
               Each element can be int/float/string.
               Here n is the number data instances in the node, p is the number of attributes.
            Y: the class labels, a numpy array of length n.
               Each element can be int/float/string.
            i: the index of the attribute being tested in the node, an integer scalar 
            C: the dictionary of attribute values and children nodes. 
               Each (key, value) pair represents an attribute value and its corresponding child node.
            isleaf: whether or not this node is a leaf node, a boolean scalar
            p: the label to be predicted on the node (i.e., most common label in the node).
    '''

    def __init__(self, X, Y, i=None, C=None, isleaf=False, p=None):
        self.X = X
        self.Y = Y
        self.i = i
        self.C = C
        self.isleaf = isleaf
        self.p = p


# -----------------------------------------------
class Tree(object):
    '''
        Decision Tree (with discrete attributes). 
        We are using ID3(Iterative Dichotomiser 3) algorithm. So this decision tree is also called ID3.
    '''

    # --------------------------
    @staticmethod
    def entropy(Y):
        '''
            Compute the entropy of a list of values.
            Input:
                Y: a list of values, a numpy array of int/float/string values.
            Output:
                e: the entropy of the list of values, a float scalar
            Hint: you could use collections.Counter.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        c = dict(Counter(Y))
        e = 0
        n = len(Y)
        for v in c.values():
            p = float(v) / n
            e = e - p * math.log(p, 2)

        #########################################
        return e

        # --------------------------

    @staticmethod
    def conditional_entropy(Y, X):
        '''
            Compute the conditional entropy of y given x.
            Input:
                Y: a list of values, a numpy array of int/float/string values.
                X: a list of values, a numpy array of int/float/string values.
            Output:
                ce: the conditional entropy of y given x, a float scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        x_counter = dict(Counter(X))
        ce = 0
        n = len(X)
        for k, v in x_counter.iteritems():
            p = float(v) / n
            cy_list = []
            for i in xrange(n):
                if X[i] == k:
                    cy_list.append(Y[i])
            cy_counter = dict(Counter(cy_list))
            ny = len(cy_list)
            for vy in cy_counter.values():
                py = float(vy) / ny
                ce = ce - p * py * math.log(py, 2)
        #########################################
        return ce

        # --------------------------

    @staticmethod
    def information_gain(Y, X):
        '''
            Compute the information gain of y after spliting over attribute x
            Input:
                X: a list of values, a numpy array of int/float/string values.
                Y: a list of values, a numpy array of int/float/string values.
            Output:
                g: the information gain of y after spliting over x, a float scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        g = Tree.entropy(Y) - Tree.conditional_entropy(Y, X)
        #########################################
        return g

    # --------------------------
    @staticmethod
    def best_attribute(X, Y):
        '''
            Find the best attribute to split the node. 
            Here we use information gain to evaluate the attributes. 
            If there is a tie in the best attributes, select the one with the smallest index.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
            Output:
                i: the index of the attribute to split, an integer scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        p, n = X.shape
        res = []
        for j in xrange(p):
            res.append(Tree.information_gain(Y, X[j]))
        i = np.argmax(res)
        #########################################
        return i

    # --------------------------
    @staticmethod
    def split(X, Y, i):
        '''
            Split the node based upon the i-th attribute.
            (1) split the matrix X based upon the values in i-th attribute
            (2) split the labels Y based upon the values in i-th attribute
            (3) build children nodes by assigning a submatrix of X and Y to each node
            (4) build the dictionary to combine each  value in the i-th attribute with a child node.
    
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
                i: the index of the attribute to split, an integer scalar
            Output:
                C: the dictionary of attribute values and children nodes. 
                   Each (key, value) pair represents an attribute value and its corresponding child node.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        p, n = X.shape
        count_i = dict(Counter(X[i]))
        C = dict()
        for k, v in count_i.iteritems():
            node_X = []
            node_Y = []
            for j in xrange(n):
                if X[i, j] == k:
                    node_X.append(X[:, j])
                    node_Y.append(Y[j])
            node_X = np.asarray(node_X).T
            node_Y = np.asarray(node_Y)
            C[k] = Node(node_X, node_Y)

        #########################################
        return C

    # --------------------------
    @staticmethod
    def stop1(Y):
        '''
            Test condition 1 (stop splitting): whether or not all the instances have the same label. 
    
            Input:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
            Output:
                s: whether or not Conidtion 1 holds, a boolean scalar. 
                True if all labels are the same. Otherwise, false.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        if len(set(Y)) == 1:
            s = True
        else:
            s = False
        #########################################
        return s

    # --------------------------
    @staticmethod
    def stop2(X):
        '''
            Test condition 2 (stop splitting): whether or not all the instances have the same attributes. 
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
            Output:
                s: whether or not Conidtion 2 holds, a boolean scalar. 
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        n = X.shape[1]
        s = True
        for i in xrange(n):
            if not np.array_equal(X[:, 0], X[:, i]):
                return False
        #########################################
        return s

    # --------------------------
    @staticmethod
    def most_common(Y):
        '''
            Get the most-common label from the list Y. 
            Input:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node.
            Output:
                y: the most common label, a scalar, can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        y = Counter(Y).most_common(1)[0][0]
        #########################################
        return y

    # --------------------------
    @staticmethod
    def build_tree(t):
        '''
            Recursively build tree nodes.
            Input:
                t: a node of the decision tree, without the subtree built.
                t.X: the feature matrix, a numpy float matrix of shape n by p.
                   Each element can be int/float/string.
                    Here n is the number data instances, p is the number of attributes.
                t.Y: the class labels of the instances in the node, a numpy array of length n.
                t.C: the dictionary of attribute values and children nodes. 
                   Each (key, value) pair represents an attribute value and its corresponding child node.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        t.p = Tree.most_common(t.Y)
        # if Condition 1 or 2 holds, stop recursion 
        if Tree.stop1(t.Y) or Tree.stop2(t.X):
            t.isleaf = True
            return
        # find the best attribute to split
        t.i = Tree.best_attribute(t.X, t.Y)
        t.C = Tree.split(t.X, t.Y, t.i)
        # recursively build subtree on each child node
        for v in t.C.values():
            Tree.build_tree(v)
        #########################################

    # --------------------------
    @staticmethod
    def train(X, Y):
        '''
            Given a training set, train a decision tree. 
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the training set, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
            Output:
                t: the root of the tree.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        t = Node(X, Y)
        Tree.build_tree(t)
        #########################################
        return t

    # --------------------------
    @staticmethod
    def inference(t, x):
        '''
            Given a decision tree and one data instance, infer the label of the instance recursively. 
            Input:
                t: the root of the tree.
                x: the attribute vector, a numpy vectr of shape p.
                   Each attribute value can be int/float/string.
            Output:
                y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        if t.isleaf:
            y = t.p
        elif x[t.i] in t.C.keys():
            y = Tree.inference(t.C[x[t.i]], x)
        else:
            y = t.p
        #########################################
        return y

    # --------------------------
    @staticmethod
    def predict(t, X):
        '''
            Given a decision tree and a dataset, predict the labels on the dataset. 
            Input:
                t: the root of the tree.
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the dataset, p is the number of attributes.
            Output:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        n = X.shape[1]
        Y = []
        for i in xrange(n):
            Y.append(Tree.inference(t, X[:, i]))
        Y = np.asarray(Y)
        #########################################
        return Y

    # --------------------------
    @staticmethod
    def load_dataset(filename='data1.csv'):
        '''
            Load dataset 1 from the CSV file: 'data1.csv'. 
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
        data = np.genfromtxt(filename, dtype='string', delimiter=',', skip_header=1)
        Y = data[:, 0]
        X = data[:, 1:].T
        #########################################
        return X, Y
