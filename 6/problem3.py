import math
import numpy as np
from collections import Counter

# Note: please don't add any new package, you should solve this problem using only the packages above.
# -------------------------------------------------------------------------
'''
    Problem 3: Hidden Markov Model (HMM)
    In this problem, you will implement a hidden markov model. 
    You could test the correctness of your code by typing `nosetests -v test3.py` in the terminal.
    Note: please don't use any existing package for HMM, implement your own version.
    Hint: For details of HMM, please read: http://cs.rochester.edu/u/james/CSC248/Lec11.pdf
'''


# -----------------------------------------------
def forward_prob(Ev, I, T, Em):
    '''
        Given a HMM and a sequence of evidence, compute the forward probability of the last hidden state (X_{n-1}) using Forward Algorithm: P(X_{n-1}, e0, e1, ..., e_{n-1}), return the state of Xt with the maximium probability. Here e0 represents the observed evidence at step 0. n is the number of steps in total.
        Input:
            Ev: the observed evidence sequence, an integer numpy vector of length n. 
                Here n is the number of time steps. Each X[i] = 0,1, ..., or p-1. 
                p is the number of possible values of the evidence variable, an integer scalar.
            I : the initial probability distribution of the hidden variable, a float vector of length c. 
                c is the number of possible values of the hidden variable, an integer scalar.
            T : the transition probablity of the hidden variable, a float numpy matrix of shape c by c
                T[i,j] represents the probability of P(Xt = j | Xt-1 = i).
            Em: the emission probability of the evidence variable, a float numpy matrix of shape c by p.
                Em[i,j] represents the probability of P(Et = j | Xt = i)
        Output:
            X: the most likely state of the hidden variable in the last time step, an integer scalar (value = 0, 1, ..., or, c-1)
            a: the forward probabilities at the last time step, a float vector of length c.
                a[i] = P(X_{n-1} = i, e0, e1, ..., e_{n-1})
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    n = Ev.shape[0]
    c = I.shape[0]

    a_prev = [I[i] * Em[i, Ev[0]] for i in xrange(c)]
    a_prev = np.asarray(a_prev)

    for i in xrange(1, n):
        a_new = [Em[k, Ev[i]] * sum([a_prev[j] * T[j, k] for j in xrange(c)]) for k in xrange(c)]
        a_prev = a_new

    a = np.copy(a_prev)
    X = np.argmax(a)
    #########################################
    return X, a


# -----------------------------------------------
def backward_prob(Ev, T, Em):
    '''
        Given a HMM and a sequence of evidence, compute the backward probability of the first hidden state (X_0) using Backward Algorithm: P(e1, ..., e_{n-1}| X_0 = i), return the state of X-0 with the maximium probability. Here e0 represents the observed evidence at step 0. n is the number of steps in total.
        Input:
            Ev: the observed evidence sequence, an integer numpy vector of length n. 
                Here n is the number of time steps. Each X[i] = 0,1, ..., or p-1. 
                p is the number of possible values of the evidence variable, an integer scalar.
            T : the transition probablity of the hidden variable, a float numpy matrix of shape c by c
                T[i,j] represents the probability of P(Xt = j | Xt-1 = i).
            Em: the emission probability of the evidence variable, a float numpy matrix of shape c by p.
                Em[i,j] represents the probability of P(Et = j | Xt = i)
        Output:
            X: the most likely state of the hidden variable in the first time step, an integer scalar (value = 0, 1, ..., or, c-1)
            b: the backward probabilities at the first time step, a float vector of length c.
                b[i] = P(e1, e2, ..., e_{n-1} | X_0 = i)
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    n = Ev.shape[0]
    c = T.shape[0]

    b_prev = np.ones(c)
    for i in xrange(n - 2, -1, -1):
        b_new = [sum([b_prev[j] * Em[j, Ev[i + 1]] * T[k, j] for j in xrange(c)]) for k in xrange(c)]
        b_prev = b_new

    b = np.copy(b_prev)
    X = np.argmax(b)
    #########################################
    return X, b


# -----------------------------------------------
def forward_backward_prob(Ev, I, T, Em, i):
    '''
        Given a HMM and a sequence of evidence, compute the forward-backward probability of the i-th hidden state (X_i) using Forward-Backward Algorithm: P(X_i, e0, e1, ..., e_{n-1}), return the state of Xi with the maximium probability. Here e0 represents the observed evidence at step 0. n is the number of steps in total.
        Input:
            Ev: the observed evidence sequence, an integer numpy vector of length n. 
                Here n is the number of time steps. Each X[i] = 0,1, ..., or p-1. 
                p is the number of possible values of the evidence variable, an integer scalar.
            I : the initial probability distribution of the hidden variable, a float vector of length c. 
                c is the number of possible values of the hidden variable, an integer scalar.
            T : the transition probablity of the hidden variable, a float numpy matrix of shape c by c
                T[i,j] represents the probability of P(Xt = j | Xt-1 = i).
            Em: the emission probability of the evidence variable, a float numpy matrix of shape c by p.
                Em[i,j] represents the probability of P(Et = j | Xt = i)
            i : the target time step to etimate probability of hidden state (at step i), an integer scalar of value 0, 1, ..., or n-1
        Output:
            X: the most likely state of the hidden variable at the i-th time step, an integer scalar (value = 0, 1, ..., or, c-1)
            p: the forward-backward probabilities at the i-th time step, a float vector of length c.
                p[k] = P(X_i = k, e0, e1, ..., e_{n-1})
            a: the forward probabilities at the i-th time step, a float vector of length c.
                a[k] = P(X_i = k, e0, e1, ..., e_{i})
            b: the backward probabilities at the i-th time step, a float vector of length c.
                b[k] = P(e_{i+1}, ..., e_{n-1} | X_i = k)
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    a = forward_prob(Ev[:i + 1], I, T, Em)[1]
    b = backward_prob(Ev[i:], T, Em)[1]
    p = a * b
    X = np.argmax(p)
    #########################################
    return X, p, a, b


# -----------------------------------------------
def most_probable_pass(Ev, I, T, Em):
    '''
        Given a HMM and a sequence of evidence, compute the most probable path of the hidden states using Viterbi Algorithm.
        Input:
            Ev: the observed evidence sequence, an integer numpy vector of length n. 
                Here n is the number of time steps. Each X[i] = 0,1, ..., or p-1. 
                p is the number of possible values of the evidence variable, an integer scalar.
            I : the initial probability distribution of the hidden variable, a float vector of length c. 
                c is the number of possible values of the hidden variable, an integer scalar.
            T : the transition probablity of the hidden variable, a float numpy matrix of shape c by c
                T[i,j] represents the probability of P(Xt = j | Xt-1 = i).
            Em: the emission probability of the evidence variable, a float numpy matrix of shape c by p.
                Em[i,j] represents the probability of P(Et = j | Xt = i)
        Output:
            X: the hidden state trajectory with maximum joint probability, an integer vector of length n.
                X[i] represents the hidden state value of the i-th step in the most probable path.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    n = Ev.shape[0]
    X = np.zeros(n)
    for i in xrange(n):
        X[i] = forward_backward_prob(Ev, I, T, Em, i)[0]
    #########################################
    return X


# -----------------------------------------------
def compute_gamma(Ev, I, T, Em):
    '''
        Given a HMM and a sequence of evidence, estimate the parameter of HMM with the highest likelihood of observing the sequence using EM (Baum-Welch) Algorithm. Compute the gamma values for the E-step of EM algorithm.
        Input:
            Ev: the observed evidence sequence, an integer numpy vector of length n. 
                Here n is the number of time steps. Each X[i] = 0,1, ..., or p-1. 
            I : the current estimation of initial probability, a float vector of length c. 
            T : the current estimation of transition probablity, a float numpy matrix of shape c by c
                T[i,j] represents the probability of P(Xt = j | Xt-1 = i).
            Em: the current estimation of emission probability, a float numpy matrix of shape c by p.
                Em[i,j] represents the probability of P(Et = j | Xt = i)
        Output:
            gamma: the gamma probability, a numpy matrix of shape n by c
                    gamma[t,i] denotes P(X_t =  i | Ev, HMM_parameters)
            alpha: the alpha probabilities at each time step, a numpy matrix of shape n by c
                    alpha[t] denotes the forward probability at time step t.
            beta: the beta probabilities at each time step, a numpy matrix of shape n by c
                    beta[t] denotes the backward probability at time step t.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    n = Ev.shape[0]
    c = I.shape[0]
    gamma = np.zeros(shape=(n, c))
    alpha = np.zeros(shape=(n, c))
    beta = np.zeros(shape=(n, c))
    for i in xrange(n):
        X, p, a, b = forward_backward_prob(Ev, I, T, Em, i)
        alpha[i] = a
        beta[i] = b
        gamma[i] = (a * b) / sum(a * b)
    #########################################
    return gamma, alpha, beta


# -----------------------------------------------
def compute_xi(Ev, T, Em, alpha, beta):
    '''
        Given a HMM and a sequence of evidence, estimate the parameter of HMM with the highest likelihood of observing the sequence using EM (Baum-Welch) Algorithm. Compute the xi values for the E-step of EM algorithm.
        Input:
            Ev: the observed evidence sequence, an integer numpy vector of length n. 
                Here n is the number of time steps. Each X[i] = 0,1, ..., or p-1. 
            I : the current estimation of initial probability, a float vector of length c. 
            T : the current estimation of transition probablity, a float numpy matrix of shape c by c
                T[i,j] represents the probability of P(Xt = j | Xt-1 = i).
            Em: the current estimation of emission probability, a float numpy matrix of shape c by p.
                Em[i,j] represents the probability of P(Et = j | Xt = i)
            alpha: the alpha probabilities at each time step, a numpy matrix of shape n by c
                    alpha[t] denotes the forward probability at time step t.
            beta: the beta probabilities at each time step, a numpy matrix of shape n by c
                    beta[t] denotes the backward probability at time step t.
        Output:
            xi: the xi probability, a numpy matrix of shape n-1 by c by c
                    xi[t,i,j] denotes P(X_t =  i, X_{t+1} = j | Ev, HMM_parameters)
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    n = Ev.shape[0]
    c = T.shape[0]
    xi = np.zeros(shape=(n - 1, c, c))
    for t in xrange(n - 1):
        sum_ = sum([alpha[t, i] * T[i, j] * beta[t + 1, j] * Em[j, Ev[t + 1]] for i in xrange(c) for j in xrange(c)])
        for i in xrange(c):
            for j in xrange(c):
                xi[t, i, j] = alpha[t, i] * beta[t + 1, j] * T[i, j] * Em[j, Ev[t + 1]] / sum_
    #########################################
    return xi


# -----------------------------------------------
def E_step(Ev, I, T, Em):
    '''
        Given a HMM and a sequence of evidence, estimate the parameter of HMM with the highest likelihood of observing the sequence using EM (Baum-Welch) Algorithm. This function is the E-step of EM algorithm.
        Input:
            Ev: the observed evidence sequence, an integer numpy vector of length n. 
                Here n is the number of time steps. Each X[i] = 0,1, ..., or p-1. 
            I : the current estimation of initial probability, a float vector of length c. 
            T : the current estimation of transition probablity, a float numpy matrix of shape c by c
                T[i,j] represents the probability of P(Xt = j | Xt-1 = i).
            Em: the current estimation of emission probability, a float numpy matrix of shape c by p.
                Em[i,j] represents the probability of P(Et = j | Xt = i)
        Output:
            gamma: the gamma probability, a numpy matrix of shape n by c
                    gamma[t,i] denotes P(X_t =  i | Ev, HMM_parameters)
            xi: the xi probability, a numpy matrix of shape n-1 by c by c
                    xi[t,i,j] denotes P(X_t =  i, X_{t+1} = j | Ev, HMM_parameters)
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    gamma, alpha, beta = compute_gamma(Ev, I, T, Em)
    xi = compute_xi(Ev, T, Em, alpha, beta)
    #########################################
    return gamma, xi


# -----------------------------------------------
def update_I(gamma):
    '''
        In the M-step of EM algorithm, update initial probabilities with gamma values.
        Input:
            gamma: the gamma probability, a numpy matrix of shape n by c
                    gamma[t,i] denotes P(X_t =  i | Ev, HMM_parameters)
        Output:
            I : the new estimation of initial probability, a float vector of length c. 
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    I = gamma[0]
    #########################################
    return I


# -----------------------------------------------
def update_T(gamma, xi):
    '''
        In the M-step of EM algorithm, update transition probabilities with gamma values.
        Input:
            gamma: the gamma probability, a numpy matrix of shape n by c
                    gamma[t,i] denotes P(X_t =  i | Ev, HMM_parameters)
            xi: the xi probability, a numpy matrix of shape n-1 by c by c
                    xi[t,i,j] denotes P(X_t =  i, X_{t+1} = j | Ev, HMM_parameters)
        Output:
            T : the updated estimation of transition probablity, a float numpy matrix of shape c by c
                T[i,j] represents the probability of P(Xt = j | Xt-1 = i).
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    n, c = gamma.shape
    T = np.zeros(shape=(c, c))
    for i in xrange(c):
        T[i] = sum([xi[t, i, :] for t in xrange(n - 1)]) / sum([gamma[t, i] for t in xrange(n)])
    s = [sum(T[i]) for i in xrange(c)]
    for i in xrange(c):
        T[i] = T[i] / s[i]
    #########################################
    return T


# -----------------------------------------------
def update_Em(Ev, gamma, p):
    '''
        In the M-step of EM algorithm, update emission probabilities with gamma values.
        Input:
            Ev: the observed evidence sequence, an integer numpy vector of length n. 
                Here n is the number of time steps. Each X[i] = 0,1, ..., or p-1. 
            gamma: the gamma probability, a numpy matrix of shape n by c
                    gamma[t,i] denotes P(X_t =  i | Ev, HMM_parameters)
            p: the number of possible values of the evidence variable, an integer scalar.
        Output:
            Em: the new estimation of emission probability, a float numpy matrix of shape c by p.
                Em[i,j] represents the probability of P(Et = j | Xt = i)
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    n, c = gamma.shape
    Em = np.zeros(shape=(c, p))
    for i in xrange(c):
        sum_ = sum(gamma[:, i])
        for j in xrange(p):
            e = 0.
            for k in xrange(n):
                if Ev[k] == j:
                    e += gamma[k, i]
            Em[i, j] = e / sum_
    #########################################
    return Em


# -----------------------------------------------
def M_step(Ev, gamma, xi, p):
    '''
        Given a HMM and a sequence of evidence, estimate the parameter of HMM with the highest likelihood of observing the sequence using EM (Baum-Welch) Algorithm. This function is the M-step of EM algorithm.
        Input:
            Ev: the observed evidence sequence, an integer numpy vector of length n. 
                Here n is the number of time steps. Each X[i] = 0,1, ..., or p-1. 
            gamma: the gamma probability, a numpy matrix of shape n by c
                    gamma[t,i] denotes P(X_t =  i | Ev, HMM_parameters)
            xi: the xi probability, a numpy matrix of shape n-1 by c by c
                    xi[t,i,j] denotes P(X_t =  i, X_{t+1} = j | Ev, HMM_parameters)
            p: the number of possible values of the evidence variable, an integer scalar.
        Output:
            I : the new estimation of initial probability, a float vector of length c. 
            T : the new estimation of transition probablity, a float numpy matrix of shape c by c
                T[i,j] represents the probability of P(Xt = j | Xt-1 = i).
            Em: the new estimation of emission probability, a float numpy matrix of shape c by p.
                Em[i,j] represents the probability of P(Et = j | Xt = i)
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    I = update_I(gamma)
    T = update_T(gamma, xi)
    Em = update_Em(Ev, gamma, p)
    #########################################
    return I, T, Em


# -----------------------------------------------
def EM(Ev, c, p, num_iter=10):
    '''
        Given a HMM and a sequence of evidence, estimate the parameter of HMM with the highest likelihood of observing the sequence using EM (Baum-Welch) Algorithm.
        Input:
            Ev: the observed evidence sequence, an integer numpy vector of length n. 
                Here n is the number of time steps. Each X[i] = 0,1, ..., or p-1. 
            c: the number of possible values of the hidden variable, an integer scalar.
            p: the number of possible values of the evidence variable, an integer scalar.
            num_iter: the number of iterations in EM, an integer scalar.
        Output:
            I : the initial probability distribution of the hidden variable, a float vector of length c. 
            T : the transition probablity of the hidden variable, a float numpy matrix of shape c by c
                T[i,j] represents the probability of P(Xt = j | Xt-1 = i).
            Em: the emission probability of the evidence variable, a float numpy matrix of shape c by p.
                Em[i,j] represents the probability of P(Et = j | Xt = i)
    '''

    # initialize parameters 
    # (This is for testing purpose. In real-world cases, we should randomly initialize the parameters.)
    I = np.arange(float(c)) + 1.
    I = I / sum(I)
    T = np.arange(float(c * c)).reshape((c, c)) + 1.
    for i in xrange(c):
        T[i] = T[i] / sum(T[i])
    Em = np.ones((c, p)) / p

    #########################################
    ## INSERT YOUR CODE HERE
    for _ in xrange(num_iter):
        gamma, xi = E_step(Ev, I, T, Em)
        I, T, Em = M_step(Ev, gamma, xi, p)
    #########################################
    return I, T, Em
