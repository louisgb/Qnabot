
import numpy as np
from scipy import optimize as scipy_optimize
import nn_aux as aux

class gbNN(object):
    """A feedforward neural network class. 

    Currently only one hidden layer. 
    Translated from MATLAB/Octave from 
      Ng's Machine Learning in Coursera with my codes filled in.
    """

    def __init__(self, size_in, size_hid, size_out):
        """Constructor. 

        size_in: input layer size
        size_hid: hidden layer size
        size_out: output layer size
        """
        self.size_in = size_in
        self.size_hid = size_hid
        self.size_out = size_out
        # self.regul = None
        #-- parameters, Theta1 and Theta2, as an array
        self.nn_params = None
        #-- 
        # self.X = None
        # self.y = None


    def nn_cost(self, X, y, regul=0.0, params=None):
        """Neural network cost function and its gradient. 

        X: numpy matrix of size m*size_in, where m=# of training data
        y: numpy array of size m, each being the label of training data
        regul: regularization parameter lambda
        params: NN parameters. Use self.nn_params if not given.
        Return:
          J: (float) total cost
          grad: (float array) gradient of J, same size as nn_params
        """

        regul = float(regul)
        if params is None: params = self.nn_params
        #-- consistency check
        if np.size(X, 0)!=np.size(y) \
           or np.size(X, 1)!=self.size_in \
           or np.size(params)!=(self.size_in+1)*self.size_hid\
                             + (self.size_hid+1)*self.size_out :
            raise ValueError('Incosistency detected in array dimensions')

        #-- Reshape nn_params back into the parameters Theta1 and Theta2
        Theta1 = np.reshape(
            params[:self.size_hid*(self.size_in+1)],
            (self.size_hid, self.size_in+1) )
        Theta2 = np.reshape(
            params[self.size_hid*(self.size_in+1):],
            (self.size_out, self.size_hid+1) )
        #-- number of training data
        m = np.size(X, 0)
        #-- append a column of 1 (x0) to X and transpose
        a1 = np.append(np.ones((m,1)), X, 1)
        a1 = np.transpose(a1)
        #-- feedforward propagate to get h(k, i) 
        #--   where k~output unit, i~training example
        z2 = np.dot(Theta1, a1)
        a2 = aux.sigmoid(z2)
        a2 = np.append(np.ones((1,np.size(a2, 1))), a2, 0)
        z3 = np.dot(Theta2, a2)
        h = aux.sigmoid(z3)
        #-- convert y to format comparable to h
        yy = np.zeros((self.size_out, m))
        for i in xrange(m):
            yy[y[i]-1,i] = 1
        #-- calculate cost without regularization
        J = -yy*np.log(h)-(1.0-yy)*np.log(1.0-h)
        J = np.sum(J)/m
        #-- add regularization; 
        #--   note Theta(:,1) are bias params not to be regularized
        J += regul/2.0/m*(
            np.sum(Theta1[:,1:]**2) + np.sum(Theta2[:,1:]**2) )

        #=====
        #=== Backpropagate for gradient
        #=====

        Theta1_grad = np.zeros_like(Theta1)
        Theta2_grad = np.zeros_like(Theta2)
        delta3 = h-yy
        #-- ignore bias units when calculating delta
        delta2 = np.dot(np.transpose(Theta2[:,1:]), delta3)\
                 *aux.sigmoid_grad(z2)
        for i in xrange(m):
            Theta1_grad += np.outer(delta2[:,i], a1[:,i])
            Theta2_grad += np.outer(delta3[:,i], a2[:,i])
        Theta1_grad /= m
        Theta2_grad /= m
        #-- add regularization
        Theta1_grad[:,1:] += Theta1[:,1:]*regul/m
        Theta2_grad[:,1:] += Theta2[:,1:]*regul/m
        # Theta1_grad[:,1:] *= 1.0+regul/m
        # Theta2_grad[:,1:] *= 1.0+regul/m

        #=====
        #=== Finalize
        #=====

        #-- unroll gradients
        grad = np.append(Theta1_grad, Theta2_grad)
        return J, grad

    def rand_init_params(self):
        """Randomly initialize Theta matrices

        Return: random initial params
        """
        eps = (6.0/(self.size_in+self.size_hid))**0.5
        theta1 = np.random.rand((self.size_in+1)*self.size_hid)\
                 *2*eps-eps
        eps = (6.0/(self.size_out+self.size_hid))**0.5
        theta2 = np.random.rand((self.size_hid+1)*self.size_out)\
                 *2*eps-eps
        return np.append(theta1, theta2)

    def load_params(self, params):
        """Load NN parameters from external.

        params: list containing parameters
        """
        size_params = (self.size_in+1)*self.size_hid\
                      + (self.size_hid+1)*self.size_out
        if np.size(params)!=size_params:
            raise ValueError('''
            Length of argument not matching %d
            '''%size_params)
        else:
            self.nn_params = np.reshape(params, -1)

    def train(self, X, y, regul=0.0, maxiter=100, initguess=None, 
              verbose=False):
        """Train parameters.

        """
        if initguess is None:
            params0 = self.rand_init_params()
        else:
            params0 = initguess
        func = lambda x:self.nn_cost(X, y, regul, x)
        optres = scipy_optimize.minimize(
            func, params0, jac=True, method='L-BFGS-B', 
            options={'maxiter':maxiter, 'disp':verbose})
        # optres = scipy_optimize.minimize(
        #     func, params0, jac=None, method='Nelder-Mead', 
        #     options={'maxiter':1, 'disp':True, 'maxfev':1})
        if not optres.success \
           and 'EXCEEDS LIMIT' not in optres.message:
            raise RuntimeWarning('''
            Training failed for reason: %s
            '''%optres.message)

        #-- save results
        self.nn_params = optres.x

        #-- print debug info
        if verbose:
            yp = self.predict(X)
            print 'Training accuracy: %.2f%%'\
                %(np.mean(np.array(yp==y, dtype=float))*100)




    def predict_raw(self, X):
        """Get vector output from input X. 

        X: numpy matrix of size m*size_in, where m=# of training data
        Return: vector output h of size output_dim*m
        """
        #-- Reshape nn_params back into the parameters Theta1 and Theta2
        Theta1 = np.reshape(
            self.nn_params[:self.size_hid*(self.size_in+1)],
            (self.size_hid, self.size_in+1) )
        Theta2 = np.reshape(
            self.nn_params[self.size_hid*(self.size_in+1):],
            (self.size_out, self.size_hid+1) )
        #== following is copy from nn_cost
        #-- number of training data
        m = np.size(X, 0)
        #-- append a column of 1 (x0) to X and transpose
        a1 = np.append(np.ones((m,1)), X, 1)
        a1 = np.transpose(a1)
        #-- feedforward propagate to get h(k, i) 
        #--   where k~output unit, i~training example
        z2 = np.dot(Theta1, a1)
        a2 = aux.sigmoid(z2)
        a2 = np.append(np.ones((1,np.size(a2, 1))), a2, 0)
        z3 = np.dot(Theta2, a2)
        h = aux.sigmoid(z3)
        return h

    def predict_ranked(self, X, num):
        """Predict the best num y from X.

        X: numpy matrix of size m*size_in, where m=# of training data
        num: number of candidate labels to return for each row of X
        Return: num*m matrix y, where y[i][j] = i-ranked candidate
                for input j
        """
        if num > self.size_out: num = self.size_out
        h = self.predict_raw(X)
        y = []
        for i in xrange(h.shape[1]):
            labels = sorted(range(self.size_out), 
                            key=lambda x: h[x][i], reverse=True)
            y.append(labels[:num])
        return np.transpose(y)

    def predict(self, X):
        """Predict best y from X. 

        X: numpy matrix of size m*size_in, where m=# of training data
        Return: vector y, composed of one label for each row of X
                (label is 1-based)
        """
        h = self.predict_raw(X)
        return np.argmax(h, 0)
        
