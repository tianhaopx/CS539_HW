import math
import numpy as np
from problem2 import DT, Node

# -------------------------------------------------------------------------
'''
    Problem 5: Boosting (on continous attributes). 
               We will implement AdaBoost algorithm in this problem.
    You could test the correctness of your code by typing `nosetests -v test5.py` in the terminal.
'''


# -----------------------------------------------
class DS(DT):
    '''
        Decision Stump (with contineous attributes) for Boosting.
        Decision Stump is also called 1-level decision tree.
        Different from other decision trees, a decision stump can have at most one level of child nodes.
        In order to be used by boosting, here we assume that the data instances are weighted.
    '''

    # --------------------------
    @staticmethod
    def entropy(Y, D):
        '''
            Compute the entropy of the weighted instances.
            Input:
                Y: a list of labels of the instances, a numpy array of int/float/string values.
                D: the weights of instances, a numpy float vector of length n
            Output:
                e: the entropy of the weighted samples, a float scalar
            Hint: you could use np.unique(). 
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        d = dict()
        for i in xrange(len(Y)):
            key = Y[i]
            if key not in d.keys():
                d[key] = 0
            d[key] += D[i]
        e = 0
        for v in d.values():
            if v == 0:
                continue
            e = e - v * math.log(v, 2)

        #########################################
        return e

        # --------------------------

    @staticmethod
    def conditional_entropy(Y, X, D):
        '''
            Compute the conditional entropy of y given x on weighted instances
            Input:
                Y: a list of values, a numpy array of int/float/string values.
                X: a list of values, a numpy array of int/float/string values.
                D: the weights of instances, a numpy float vector of length n
            Output:
                ce: the weighted conditional entropy of y given x, a float scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        px_dict = dict()
        for i in xrange(len(X)):
            px_dict[X[i]] = px_dict.get(X[i], 0) + D[i]
        ce = 0
        for key, val in px_dict.items():
            if val == 0:
                continue
            py_dict = dict()
            for i in xrange(len(X)):
                if X[i] == key:
                    py_dict[Y[i]] = py_dict.get(Y[i], 0) + D[i] / float(val)
            for val_y in py_dict.values():
                if val_y == 0:
                    continue
                ce -= val * val_y * math.log(val_y, 2)

        #########################################
        return ce

        # --------------------------

    @staticmethod
    def information_gain(Y, X, D):
        '''
            Compute the information gain of y after spliting over attribute x
            Input:
                X: a list of values, a numpy array of int/float/string values.
                Y: a list of values, a numpy array of int/float/string values.
                D: the weights of instances, a numpy float vector of length n
            Output:
                g: the weighted information gain of y after spliting over x, a float scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        g = DS.entropy(Y, D) - DS.conditional_entropy(Y, X, D)
        #########################################
        return g

    # --------------------------
    @staticmethod
    def best_threshold(X, Y, D):
        '''
            Find the best threshold among all possible cutting points in the continous attribute of X. The data instances are weighted. 
            Input:
                X: a list of values, a numpy array of int/float values.
                Y: a list of values, a numpy array of int/float/string values.
                D: the weights of instances, a numpy float vector of length n
            Output:
            Output:
                th: the best threhold, a float scalar. 
                g: the weighted information gain by using the best threhold, a float scalar. 
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        cp = DT.cutting_points(X, Y)
        if cp[0] == float('-inf'):
            th = float('-inf')
            g = -1
        else:
            X, Y, D = (list(t) for t in zip(*sorted(zip(X, Y, D))))
            igs = []
            for p in cp:
                newX = []
                for i in X:
                    newX.append(0) if i < p else newX.append(1)
                igs.append(DS.information_gain(Y, newX, D))
            index = np.argmax(igs)
            th = cp[index]
            g = igs[index]
        #########################################
        return th, g

        # --------------------------

    def best_attribute(self, X, Y, D):
        '''
            Find the best attribute to split the node. The attributes have continous values (int/float). The data instances are weighted.
            Here we use information gain to evaluate the attributes. 
            If there is a tie in the best attributes, select the one with the smallest index.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                D: the weights of instances, a numpy float vector of length n
            Output:
                i: the index of the attribute to split, an integer scalar
                th: the threshold of the attribute to split, a float scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        p, n = X.shape
        i = 0
        g_max = float("-inf")
        th = 0
        for j in range(p):
            x = X[j]
            local_th, g = DS.best_threshold(x, Y, D)
            if g > g_max:
                i = j
                th = local_th
                g_max = g
        #########################################
        return i, th

    # --------------------------
    @staticmethod
    def most_common(Y, D):
        '''
            Get the most-common label from the list Y. The instances are weighted.
            Input:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node.
                D: the weights of instances, a numpy float vector of length n
            Output:
                y: the most common label, a scalar, can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        d = dict()
        for i in xrange(len(Y)):
            d[Y[i]] = d.get(Y[i], 0) + D[i]
        y = DT.most_common(d)
        #########################################
        return y

    # --------------------------
    def build_tree(self, X, Y, D):
        '''
            build decision stump by overwritting the build_tree function in DT class.
            Instead of building tree nodes recursively in DT, here we only build at most one level of children nodes.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                D: the weights of instances, a numpy float vector of length n
            Return:
                t: the root node of the decision stump. 
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        t = Node(X, Y)
        t.p = DS.most_common(t.Y, D)
        # if Condition 1 or 2 holds, stop splitting 
        if DS.stop1(t.Y) or DS.stop2(t.X):
            t.isleaf = True
            return t
        # find the best attribute to split
        t.i, t.th = self.best_attribute(t.X, t.Y, D)
        # configure each child node
        t.C1, t.C2 = self.split(t.X, t.Y, t.i, t.th)
        D1 = []
        D2 = []
        for j in range(len(Y)):
            if X[t.i, j] < t.th:
                D1.append(D[j])
            else:
                D2.append(D[j])
        t.C1.isleaf = True
        t.C1.p = DS.most_common(t.C1.Y, D1)
        t.C2.isleaf = True
        t.C2.p = DS.most_common(t.C2.Y, D2)
        #########################################
        return t


# -----------------------------------------------
class AB(DS):
    '''
        AdaBoost algorithm (with contineous attributes).
    '''

    # --------------------------
    @staticmethod
    def weighted_error_rate(Y, Y_, D):
        '''
            Compute the weighted error rate of a decision on a dataset. 
            Input:
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                Y_: the predicted class labels, a numpy array of length n. Each element can be int/float/string.
                D: the weights of instances, a numpy float vector of length n
            Output:
                e: the weighted error rate of the decision stump
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        e = 0
        for i in xrange(len(Y)):
            if Y[i] != Y_[i]:
                e += D[i]
        #########################################
        return e

    # --------------------------
    @staticmethod
    def compute_alpha(e):
        '''
            Compute the weight a decision stump based upon weighted error rate.
            Input:
                e: the weighted error rate of a decision stump
            Output:
                a: (alpha) the weight of the decision stump, a float scalar.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        if e == 0:
            a = 500.
        elif e == 1:
            a = -500.
        else:
            a = 0.5 * np.log((1 - e) / e)
        #########################################
        return a

    # --------------------------
    @staticmethod
    def update_D(D, a, Y, Y_):
        '''
            update the weight the data instances 
            Input:
                D: the current weights of instances, a numpy float vector of length n
                a: (alpha) the weight of the decision stump, a float scalar.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                Y_: the predicted class labels by the decision stump, a numpy array of length n. Each element can be int/float/string.
            Output:
                D: the new weights of instances, a numpy float vector of length n
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        newD = np.zeros(len(D))
        for i in range(len(D)):
            if Y_[i] != Y[i]:
                newD[i] = D[i] * float(np.exp(a))
            else:
                newD[i] = D[i] * float(np.exp(-a))
        newD = newD / sum(newD)
        D = newD
        #########################################
        return D

    # --------------------------
    @staticmethod
    def step(X, Y, D):
        '''
            Compute one step of Boosting.  
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                D: the current weights of instances, a numpy float vector of length n
            Output:
                t:  the root node of a decision stump trained in this step
                a: (alpha) the weight of the decision stump, a float scalar.
                D: the new weights of instances, a numpy float vector of length n
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        ds = DS()
        t = ds.build_tree(X, Y, D)
        Y_ = [DS.inference(t, X[:, i]) for i in range(X.shape[1])]
        e = AB.weighted_error_rate(Y, Y_, D)
        a = AB.compute_alpha(e)
        D = AB.update_D(D, a, Y, Y_)
        #########################################
        return t, a, D

    # --------------------------
    @staticmethod
    def inference(x, T, A):
        '''
            Given a bagging ensemble of decision trees and one data instance, infer the label of the instance. 
            Input:
                x: the attribute vector of a data instance, a numpy vectr of shape p.
                   Each attribute value can be int/float
                T:  the root nodes of decision stumps, a list of length n_tree. 
                A: the weights of the decision stumps, a numpy float vector of length n_tree.
            Output:
                y: the class label, a scalar of int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        l = [DT.inference(t, x) for t in T]
        d = dict()
        for i in xrange(len(l)):
            d[l[i]] = d.get(l[i], 0) + A[i]
        y = DT.most_common(d)
        #########################################
        return y

    # --------------------------
    @staticmethod
    def predict(X, T, A):
        '''
            Given an AdaBoost and a dataset, predict the labels on the dataset. 
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                T:  the root nodes of decision stumps, a list of length n_tree. 
                A: the weights of the decision stumps, a numpy float vector of length n_tree.
            Output:
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        n = X.shape[1]
        Y = np.empty(n, dtype='object')
        for i in range(n):
            Y[i] = AB.inference(X[:, i], T, A)
        #########################################
        return Y

        # --------------------------

    @staticmethod
    def train(X, Y, n_tree=10):
        '''
            train adaboost.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                n_tree: the number of trees in the ensemble, an integer scalar
            Output:
                T:  the root nodes of decision stumps, a list of length n_tree. 
                A: the weights of the decision stumps, a numpy float vector of length n_tree.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        p, n = X.shape
        # initialize weight as 1/n
        D = np.ones(n) / n
        T = []
        A = []
        # iteratively build decision stumps
        for _ in xrange(n_tree):
            t, a, D = AB.step(X, Y, D)
            T.append(t)
            A.append(a)
        T = np.asarray(T)
        A = np.asarray(A)
        #########################################
        return T, A
